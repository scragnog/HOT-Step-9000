"""Advanced multi-adapter management with weight-space merging.

Provides slot-based multi-adapter loading (up to 4 simultaneous adapters),
per-slot scaling, and per-slot per-group scaling (self_attn, cross_attn, mlp).

Ported from custombuild handler.py — uses weight-space merging approach:
1. Backup base decoder weights to CPU on first load
2. For each adapter: load → merge → compute delta = merged − base → store on CPU
3. At inference: decoder = base + Σ(slot_scale × group_scale × delta)
"""

import gc
import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger


MAX_ADAPTER_SLOTS = 4


def _determine_group(module_name: str) -> str:
    """Determine which module group a named module belongs to.

    Key naming in the decoder's LinearTransformerBlock:
      - .cross_attn.  → cross-attention (checked first, contains 'attn')
      - .attn.        → self / joint attention
      - .ff.          → feed-forward / MLP
    """
    if "cross_attn" in module_name:
        return "cross_attn"
    elif ".attn." in module_name or ".attn_" in module_name:
        return "self_attn"
    elif ".ff." in module_name or ".ff_" in module_name:
        return "mlp"
    return ""


def _extract_layer_index(key: str) -> Optional[int]:
    """Extract transformer layer index from a weight key.

    Example: ``'layers.7.attn.qkv.weight'`` → ``7``.
    Returns ``None`` if no layer index is found in the key.
    """
    parts = key.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def _derive_adapter_name(lora_path: str, safetensors_file: Optional[str]) -> str:
    """Derive a human-readable adapter name from the path."""
    if safetensors_file:
        name = os.path.splitext(os.path.basename(safetensors_file))[0]
    else:
        name = (
            os.path.basename(os.path.dirname(lora_path))
            if not os.path.isdir(lora_path)
            else os.path.basename(lora_path)
        )
    # Avoid generic names
    if name in ("adapter", "best", "final", "lokr_weights"):
        parent = os.path.dirname(lora_path) if os.path.isdir(lora_path) else os.path.dirname(os.path.dirname(lora_path))
        name = os.path.basename(parent)
    return name


def _extract_adapter_delta(self, lora_path: str) -> dict:
    """Load adapter, compute delta = adapted − base, restore base.

    Returns:
        Dict with keys: delta, type, safetensors_file
    """
    if self._base_decoder is None:
        raise RuntimeError("Base decoder not backed up yet")

    # Determine adapter type
    is_peft = False
    lokr_weights_path = None
    if os.path.isdir(lora_path):
        if os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            is_peft = True
        st_files = sorted(glob.glob(os.path.join(lora_path, "*.safetensors")))
        if st_files:
            lokr_first = [f for f in st_files if "lokr" in os.path.basename(f).lower()]
            lokr_weights_path = lokr_first[0] if lokr_first else st_files[0]
    elif lora_path.endswith(".safetensors"):
        lokr_weights_path = lora_path

    # Reset dynamo state — nano-vllm's global config changes
    # (capture_scalar_outputs, @torch.compile decorators) contaminate
    # the compilation context, causing "Offset increment outside graph
    # capture" when we modify decoder weights.
    torch._dynamo.reset()
    self.model.decoder.load_state_dict(self._base_decoder, strict=False)
    self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
    self.model.decoder.eval()

    adapter_type = None

    if is_peft:
        from peft import PeftModel

        # Move to CPU for PEFT loading — same CUDA graph guard as LoKR below
        original_device = self.device
        self.model.decoder = self.model.decoder.cpu()
        self.model.decoder = PeftModel.from_pretrained(
            self.model.decoder, lora_path, is_trainable=False,
        )
        self.model.decoder = self.model.decoder.to(original_device).to(self.dtype)
        self.model.decoder.eval()
        self.model.decoder = self.model.decoder.merge_and_unload()
        adapter_type = "peft_lora"

    elif lokr_weights_path is not None:
        from safetensors import safe_open

        from acestep.core.generation.handler.lora.lokr_config import LoKRConfig

        try:
            from lycoris import LycorisNetwork
        except ImportError:
            raise ImportError("LyCORIS library not installed")

        # Snapshot all hook IDs on every decoder sub-module BEFORE injection
        # so we can remove only the NEW hooks that LyCORIS adds.
        pre_hooks = {}
        for name, module in self.model.decoder.named_modules():
            fwd = set(getattr(module, '_forward_hooks', {}).keys())
            pre = set(getattr(module, '_forward_pre_hooks', {}).keys())
            if fwd or pre:
                pre_hooks[name] = (fwd, pre)

        # Move model to CPU for lycoris injection
        original_device = self.device
        self.model.decoder = self.model.decoder.cpu()

        from acestep.core.generation.handler.lora.lifecycle import _load_lokr_adapter
        lycoris_net = _load_lokr_adapter(self.model.decoder, lokr_weights_path)

        self.model.decoder = self.model.decoder.to(original_device).to(self.dtype)
        self.model.decoder.eval()

        # LyCORIS uses forward hooks — merge_to() bakes effect into weights
        lycoris_net.merge_to()
        adapted_sd = {k: v.detach().cpu().clone() for k, v in self.model.decoder.state_dict().items()}
        try:
            lycoris_net.restore()
        except Exception:
            pass

        # === Critical cleanup ===
        # Remove ALL hooks that were added by inject_lokr_into_dit / apply_to().
        # LyCORIS restore() only un-bakes merge_to() weight changes but does NOT
        # remove forward hooks.  Without this cleanup, the last adapter's hooks
        # fire during inference ON TOP of the weight-space merged values.
        removed_count = 0
        for name, module in self.model.decoder.named_modules():
            old_fwd, old_pre = pre_hooks.get(name, (set(), set()))

            # Remove new forward hooks
            new_fwd = set(getattr(module, '_forward_hooks', {}).keys()) - old_fwd
            for hid in new_fwd:
                del module._forward_hooks[hid]
                removed_count += 1

            # Remove new forward pre-hooks
            new_pre = set(getattr(module, '_forward_pre_hooks', {}).keys()) - old_pre
            for hid in new_pre:
                del module._forward_pre_hooks[hid]
                removed_count += 1

        if removed_count:
            logger.info(f"Removed {removed_count} LyCORIS hooks from decoder")

        # Clean up the lycoris_net reference
        try:
            if hasattr(self.model.decoder, '_lycoris_net'):
                delattr(self.model.decoder, '_lycoris_net')
        except Exception:
            pass

        del lycoris_net
        adapter_type = "lycoris_lokr"
    else:
        raise ValueError(
            "Invalid adapter path. Expected PEFT dir (adapter_config.json) "
            "or LoKr weights (.safetensors)."
        )

    # Get adapted state_dict (for PEFT, merge already happened)
    if is_peft:
        adapted_sd = {k: v.detach().cpu().clone() for k, v in self.model.decoder.state_dict().items()}

    # Compute delta = adapted − base (CPU, only changed keys)
    delta = {}
    for k in self._base_decoder:
        if k in adapted_sd:
            diff = adapted_sd[k].float() - self._base_decoder[k].float()
            if diff.abs().max().item() > 1e-8:
                delta[k] = diff
    del adapted_sd

    # Restore decoder to base state
    self.model.decoder.load_state_dict(self._base_decoder, strict=False)
    self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
    self.model.decoder.eval()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    delta_mb = sum(v.numel() * 4 for v in delta.values()) / (1024**2)
    logger.info(f"Extracted adapter delta: {len(delta)} keys, {delta_mb:.1f}MB (fp32 on CPU)")
    return {"delta": delta, "type": adapter_type, "safetensors_file": lokr_weights_path}


def _apply_merged_weights(self) -> None:
    """Compute base + Σ(scale_i × delta_i) and load onto GPU decoder."""
    if self._base_decoder is None:
        return

    # --- Diagnostic: count hooks on decoder BEFORE merge ---
    hook_count = 0
    for m in self.model.decoder.modules():
        hook_count += len(getattr(m, '_forward_hooks', {}))
        hook_count += len(getattr(m, '_forward_pre_hooks', {}))
    logger.info(f"[DIAG] _apply_merged_weights: decoder has {hook_count} total hooks BEFORE merge")
    logger.info(f"[DIAG] use_lora={self.use_lora}, lora_loaded={getattr(self, 'lora_loaded', '?')}, "
                f"slots={list(self._adapter_slots.keys())}")

    active_slots = {
        sid: s for sid, s in self._adapter_slots.items()
        if s["scale"] > 0 and self.use_lora
    }

    if not active_slots:
        logger.info(f"[DIAG] No active slots (use_lora={self.use_lora}), restoring base decoder")
        self.model.decoder.load_state_dict(self._base_decoder, strict=False)
        self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
        self.model.decoder.eval()
        self._merged_dirty = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    # Log slot details
    for sid, s in active_slots.items():
        delta_norm = sum(v.float().norm().item() for v in s["delta"].values()) / max(len(s["delta"]), 1)
        logger.info(f"[DIAG] Slot {sid}: name={s['name']}, scale={s['scale']:.3f}, "
                    f"type={s['type']}, delta_keys={len(s['delta'])}, avg_delta_norm={delta_norm:.6f}")

    t0 = time.time()
    merged = {}
    all_keys = set()
    for s in active_slots.values():
        all_keys.update(s["delta"].keys())

    # Sample a few keys for diagnostic
    sample_keys = sorted(all_keys)[:3]

    for k in self._base_decoder:
        base_val = self._base_decoder[k]
        if k in all_keys:
            combined = base_val.float()
            for s in active_slots.values():
                if k in s["delta"]:
                    combined = combined + s["scale"] * s["delta"][k]
            merged[k] = combined.to(dtype=base_val.dtype)

            # Log a few sample keys
            if k in sample_keys:
                base_norm = base_val.float().norm().item()
                merged_norm = merged[k].float().norm().item()
                logger.info(f"[DIAG] Key '{k}': base_norm={base_norm:.4f} -> merged_norm={merged_norm:.4f}")
        else:
            merged[k] = base_val

    torch._dynamo.reset()
    self.model.decoder.load_state_dict(merged, strict=False)
    self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
    self.model.decoder.eval()

    # --- Diagnostic: count hooks on decoder AFTER merge ---
    hook_count_after = 0
    for m in self.model.decoder.modules():
        hook_count_after += len(getattr(m, '_forward_hooks', {}))
        hook_count_after += len(getattr(m, '_forward_pre_hooks', {}))
    logger.info(f"[DIAG] decoder has {hook_count_after} total hooks AFTER merge")

    del merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    slot_desc = ", ".join(
        f"slot{sid}={s['name']}@{s['scale']:.2f}" for sid, s in active_slots.items()
    )
    logger.info(f"Merged {len(active_slots)} adapter(s) in {elapsed:.1f}s: {slot_desc}")
    self._merged_dirty = False


def _apply_merged_weights_with_groups(self) -> None:
    """Like _apply_merged_weights but applies per-group scaling."""
    if self._base_decoder is None:
        return

    active_slots = {
        sid: s for sid, s in self._adapter_slots.items()
        if s["scale"] > 0 and self.use_lora
    }

    if not active_slots:
        self.model.decoder.load_state_dict(self._base_decoder, strict=False)
        self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
        self.model.decoder.eval()
        self._merged_dirty = False
        return

    t0 = time.time()
    merged = {}
    all_keys = set()
    for s in active_slots.values():
        all_keys.update(s["delta"].keys())

    for k in self._base_decoder:
        base_val = self._base_decoder[k]
        if k in all_keys:
            group = _determine_group(k)
            layer_idx = _extract_layer_index(k)
            combined = base_val.float()
            for s in active_slots.values():
                if k in s["delta"]:
                    g_scale = s.get("group_scales", {}).get(group, 1.0)
                    l_scales = s.get("layer_scales", {})
                    if layer_idx is not None:
                        l_scale = l_scales.get(layer_idx, 1.0)
                    elif l_scales:
                        l_scale = sum(l_scales.values()) / max(len(l_scales), 1)
                    else:
                        l_scale = 1.0
                    combined = combined + s["scale"] * g_scale * l_scale * s["delta"][k]
            merged[k] = combined.to(dtype=base_val.dtype)
        else:
            merged[k] = base_val

    self.model.decoder.load_state_dict(merged, strict=False)
    self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
    self.model.decoder.eval()

    del merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    logger.info(f"Merged {len(active_slots)} adapter(s) with group scales in {elapsed:.1f}s")
    self._merged_dirty = False


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def load_lora_slot(self, lora_path: str, slot: Optional[int] = None) -> str:
    """Load LoRA/LoKr adapter into a numbered slot.

    Args:
        lora_path: Path to adapter directory or .safetensors file
        slot: Optional slot ID. None = auto-assign next slot.

    Returns:
        Status message
    """
    if self.model is None:
        return "❌ Model not initialized. Please initialize service first."

    if self.quantization is not None:
        return (
            f"❌ LoRA loading is not supported on quantized models. "
            f"Current quantization: {self.quantization}. "
            "Please re-initialize with quantization disabled."
        )

    if not lora_path or not lora_path.strip():
        return "❌ Please provide a LoRA path."
    lora_path = lora_path.strip()

    if not os.path.exists(lora_path):
        return f"❌ LoRA path not found: {lora_path}"

    if len(self._adapter_slots) >= MAX_ADAPTER_SLOTS:
        return f"❌ Maximum {MAX_ADAPTER_SLOTS} adapter slots. Please unload one first."

    try:
        # Backup base decoder on first load
        if self._base_decoder is None:
            logger.info("Backing up base decoder state_dict to CPU")
            self._base_decoder = {
                k: v.detach().cpu().clone() for k, v in self.model.decoder.state_dict().items()
            }
            backup_mb = sum(v.numel() * v.element_size() for v in self._base_decoder.values()) / (1024**2)
            logger.info(f"Base decoder backed up ({backup_mb:.1f}MB)")

        # Extract delta
        logger.info(f"Extracting adapter delta from {lora_path}")
        result = _extract_adapter_delta(self, lora_path)

        # Determine slot ID
        if slot is None:
            slot = self._next_slot_id
        self._next_slot_id = max(self._next_slot_id, slot + 1)

        # Derive adapter name
        adapter_name = _derive_adapter_name(lora_path, result.get("safetensors_file"))

        self._adapter_slots[slot] = {
            "path": lora_path,
            "name": adapter_name,
            "type": result["type"],
            "delta": result["delta"],
            "scale": 1.0,
            "group_scales": {"self_attn": 1.0, "cross_attn": 1.0, "mlp": 1.0},
            "layer_scales": {},  # empty = all layers at 1.0
        }

        self.use_lora = True
        self.lora_loaded = True
        self._merged_dirty = True

        # Extract trigger word metadata from safetensors header
        safetensors_file = result.get("safetensors_file")
        if safetensors_file:
            from acestep.core.generation.handler.lora.lifecycle import _read_trigger_word_from_safetensors
            tw, tp = _read_trigger_word_from_safetensors(safetensors_file)
            if tw:
                self._adapter_trigger_word = tw
                self._adapter_tag_position = tp or "prepend"
                logger.info(f"Adapter trigger word: '{tw}' (position: {tp or 'prepend'})")
            else:
                self._adapter_trigger_word = ""
                self._adapter_tag_position = ""

        _apply_merged_weights(self)

        delta_keys = len(result["delta"])
        type_label = "LoRA" if result["type"] == "peft_lora" else "LoKr"
        logger.info(f"Adapter loaded into slot {slot}: {adapter_name} ({type_label}, {delta_keys} keys)")
        return f"✅ {type_label} loaded into slot {slot}: {adapter_name}"

    except Exception as e:
        logger.exception("Failed to load adapter")
        return f"❌ Failed to load adapter: {str(e)}"


def unload_lora_slot(self, slot: Optional[int] = None) -> str:
    """Unload adapter from a slot, or all adapters.

    Args:
        slot: Slot ID to unload. None = unload all.

    Returns:
        Status message
    """
    if not self._adapter_slots:
        return "⚠️ No adapters loaded."

    if self._base_decoder is None:
        return "❌ Base decoder backup not found."

    try:
        if slot is not None:
            if slot not in self._adapter_slots:
                return f"❌ Slot {slot} not found. Active slots: {list(self._adapter_slots.keys())}"
            name = self._adapter_slots[slot]["name"]
            del self._adapter_slots[slot]
            self._merged_dirty = True
            _apply_merged_weights(self)
            if not self._adapter_slots:
                self.use_lora = False
                self.lora_loaded = False
                self._adapter_trigger_word = ""
                self._adapter_tag_position = ""
            logger.info(f"Unloaded adapter from slot {slot}: {name}")
            return f"✅ Unloaded slot {slot}: {name}"
        else:
            count = len(self._adapter_slots)
            self._adapter_slots.clear()
            self._next_slot_id = 0
            self.use_lora = False
            self.lora_loaded = False
            self._adapter_trigger_word = ""
            self._adapter_tag_position = ""
            self._merged_dirty = True
            _apply_merged_weights(self)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded all {count} adapter(s)")
            return f"✅ Unloaded all {count} adapter(s), using base model"

    except Exception as e:
        logger.exception("Failed to unload adapter")
        return f"❌ Failed to unload: {str(e)}"


def set_use_lora_advanced(self, use_lora: bool) -> str:
    """Toggle adapter usage. When disabled, decoder uses pure base weights."""
    if use_lora and not self._adapter_slots:
        return "❌ No adapters loaded. Please load one first."

    prev = self.use_lora
    self.use_lora = use_lora

    if prev != use_lora:
        self._merged_dirty = True
        _apply_merged_weights(self)

    status = "enabled" if use_lora else "disabled"
    return f"✅ Adapters {status}"


def set_lora_slot_scale(self, scale: float, slot: Optional[int] = None) -> str:
    """Set adapter scale for a slot or all slots.

    Args:
        scale: Scale value (0.0–2.0)
        slot: Specific slot. None = all slots.
    """
    if not self._adapter_slots:
        return "⚠️ No adapters loaded"

    scale = max(0.0, min(2.0, scale))

    if slot is not None:
        if slot not in self._adapter_slots:
            return f"❌ Slot {slot} not found"
        self._adapter_slots[slot]["scale"] = scale
        self._merged_dirty = True
        _apply_merged_weights_with_groups(self)
        name = self._adapter_slots[slot]["name"]
        return f"✅ Slot {slot} ({name}) scale: {scale:.2f}"
    else:
        for s in self._adapter_slots.values():
            s["scale"] = scale
        self._merged_dirty = True
        _apply_merged_weights_with_groups(self)
        return f"✅ All adapter scales set to {scale:.2f}"


def set_lora_group_scales(
    self, self_attn_scale: float, cross_attn_scale: float, mlp_scale: float,
) -> str:
    """Set per-module-group global scales applied to all slots."""
    scales = {
        "self_attn": max(0.0, min(2.0, self_attn_scale)),
        "cross_attn": max(0.0, min(2.0, cross_attn_scale)),
        "mlp": max(0.0, min(2.0, mlp_scale)),
    }
    self.lora_group_scales = scales

    for s in self._adapter_slots.values():
        s["group_scales"] = dict(scales)

    if self._adapter_slots and self.use_lora:
        self._merged_dirty = True
        _apply_merged_weights_with_groups(self)

    sa, ca, ml = scales["self_attn"], scales["cross_attn"], scales["mlp"]
    return f"✅ Group scales (all slots): SA={sa:.0%} CA={ca:.0%} MLP={ml:.0%}"


def set_slot_group_scales(
    self, slot: int, self_attn_scale: float = 1.0,
    cross_attn_scale: float = 1.0, mlp_scale: float = 1.0,
) -> str:
    """Set per-group LoRA scales for a specific adapter slot."""
    if slot not in self._adapter_slots:
        return f"❌ Slot {slot} not found. Active slots: {list(self._adapter_slots.keys())}"

    scales = {
        "self_attn": max(0.0, min(2.0, self_attn_scale)),
        "cross_attn": max(0.0, min(2.0, cross_attn_scale)),
        "mlp": max(0.0, min(2.0, mlp_scale)),
    }
    self._adapter_slots[slot]["group_scales"] = scales

    if self.use_lora:
        self._merged_dirty = True
        _apply_merged_weights_with_groups(self)

    name = self._adapter_slots[slot]["name"]
    sa, ca, ml = scales["self_attn"], scales["cross_attn"], scales["mlp"]
    return f"✅ Slot {slot} ({name}) group scales: SA={sa:.0%} CA={ca:.0%} MLP={ml:.0%}"


def set_slot_layer_scales(self, slot: int, layer_scales: Dict[int, float]) -> str:
    """Set per-layer LoRA scales for a specific adapter slot.

    Args:
        slot: Slot ID.
        layer_scales: Dict mapping layer index (0–23) to scale (0.0–2.0).
            Unlisted layers default to 1.0.
    """
    if slot not in self._adapter_slots:
        return f"❌ Slot {slot} not found. Active slots: {list(self._adapter_slots.keys())}"

    clamped = {int(k): max(0.0, min(2.0, v)) for k, v in layer_scales.items()}
    self._adapter_slots[slot]["layer_scales"] = clamped

    if self.use_lora:
        self._merged_dirty = True
        _apply_merged_weights_with_groups(self)

    name = self._adapter_slots[slot]["name"]
    if clamped:
        desc = ", ".join(f"L{k}={v:.0%}" for k, v in sorted(clamped.items()))
    else:
        desc = "all=100%"
    return f"✅ Slot {slot} ({name}) layer scales: {desc}"


def set_slot_layer_scale(self, slot: int, layer: int, scale: float) -> str:
    """Set the scale for a single layer on a specific adapter slot.

    Args:
        slot: Slot ID.
        layer: Transformer layer index (0–23).
        scale: Scale value (0.0–2.0). Set to 1.0 to restore default.
    """
    if slot not in self._adapter_slots:
        return f"❌ Slot {slot} not found. Active slots: {list(self._adapter_slots.keys())}"

    layer = int(layer)
    scale = max(0.0, min(2.0, scale))

    l_scales = self._adapter_slots[slot].setdefault("layer_scales", {})
    if abs(scale - 1.0) < 1e-6:
        l_scales.pop(layer, None)  # Remove = restore default
    else:
        l_scales[layer] = scale

    if self.use_lora:
        self._merged_dirty = True
        _apply_merged_weights_with_groups(self)

    name = self._adapter_slots[slot]["name"]
    return f"✅ Slot {slot} ({name}) layer {layer} scale: {scale:.0%}"


def get_advanced_lora_status(self) -> Dict[str, Any]:
    """Get current advanced adapter status with slot and group details."""
    slots = []
    for sid, s in sorted(self._adapter_slots.items()):
        slots.append({
            "slot": sid,
            "name": s["name"],
            "path": s["path"],
            "type": s["type"],
            "scale": s["scale"],
            "delta_keys": len(s["delta"]),
            "group_scales": s.get("group_scales", {"self_attn": 1.0, "cross_attn": 1.0, "mlp": 1.0}),
            "layer_scales": s.get("layer_scales", {}),
        })

    return {
        "loaded": len(self._adapter_slots) > 0,
        "active": self.use_lora,
        "slots": slots,
        "group_scales": dict(self.lora_group_scales),
        "temporal_schedule_active": getattr(self, "_temporal_schedule", None) is not None,
    }


# ------------------------------------------------------------------
# Temporal adapter scheduling
# ------------------------------------------------------------------

def _apply_merged_weights_temporal(self, schedule_scales: Dict[int, float]) -> None:
    """Merge adapter deltas using per-slot scales from a temporal schedule.

    Like ``_apply_merged_weights_with_groups`` but each slot's effective scale
    is overridden by ``schedule_scales[slot_id]`` for this particular diffusion
    step.  Group scales and layer scales still apply multiplicatively on top.

    Args:
        schedule_scales: Mapping of slot ID → effective scale for this step.
    """
    if self._base_decoder is None:
        return

    active_slots = {
        sid: s for sid, s in self._adapter_slots.items()
        if schedule_scales.get(sid, 0.0) > 0 and self.use_lora
    }

    if not active_slots:
        torch._dynamo.reset()
        self.model.decoder.load_state_dict(self._base_decoder, strict=False)
        self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
        self.model.decoder.eval()
        return

    merged = {}
    all_keys = set()
    for s in active_slots.values():
        all_keys.update(s["delta"].keys())

    for k in self._base_decoder:
        base_val = self._base_decoder[k]
        if k in all_keys:
            group = _determine_group(k)
            layer_idx = _extract_layer_index(k)
            combined = base_val.float()
            for sid, s in active_slots.items():
                if k in s["delta"]:
                    slot_scale = schedule_scales.get(sid, 0.0)
                    g_scale = s.get("group_scales", {}).get(group, 1.0)
                    l_scales = s.get("layer_scales", {})
                    if layer_idx is not None:
                        l_scale = l_scales.get(layer_idx, 1.0)
                    elif l_scales:
                        # Non-layer key (norm, embedding, etc.): use average of set scales
                        # so zeroing all layers also zeroes non-layer weights
                        l_scale = sum(l_scales.values()) / max(len(l_scales), 1)
                    else:
                        l_scale = 1.0
                    combined = combined + slot_scale * g_scale * l_scale * s["delta"][k]
            merged[k] = combined.to(dtype=base_val.dtype)
        else:
            merged[k] = base_val

    torch._dynamo.reset()
    self.model.decoder.load_state_dict(merged, strict=False)
    self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
    self.model.decoder.eval()
    del merged


def set_temporal_schedule(self, schedule) -> str:
    """Set or clear the temporal adapter schedule for the next generation.

    Args:
        schedule: A :class:`TemporalAdapterSchedule` instance, or ``None``
            to clear any active schedule (return to static merge).

    Returns:
        Status message.
    """
    if schedule is None:
        self._temporal_schedule = None
        # Restore static merged weights
        if self._adapter_slots and self.use_lora:
            _apply_merged_weights_with_groups(self)
        return "✅ Temporal schedule cleared — using static adapter weights"

    issues = schedule.validate()
    if issues:
        return f"❌ Invalid schedule: {'; '.join(issues)}"

    # Check that referenced slots exist
    missing = [sid for sid in schedule.slot_segments if sid not in self._adapter_slots]
    if missing:
        return f"❌ Schedule references slots {missing} which are not loaded. Active: {list(self._adapter_slots.keys())}"

    self._temporal_schedule = schedule
    slot_desc = ", ".join(
        f"slot{sid}={len(segs)} segs" for sid, segs in schedule.slot_segments.items()
    )
    return f"✅ Temporal schedule set: {slot_desc}"


def build_temporal_step_callback(self):
    """Build a callback for the diffusion loop that re-merges weights per step.

    Returns ``None`` if no temporal schedule is active. Otherwise returns a
    callable ``callback(step_idx, t_curr, total_steps)`` that the diffusion
    loop should invoke before each decoder forward pass.
    """
    schedule = getattr(self, "_temporal_schedule", None)
    if schedule is None:
        return None

    def _on_step(step_idx: int, t_curr: float, total_steps: int):
        """Re-merge adapter weights for this diffusion step."""
        # Map diffusion step to normalised song position.
        # Diffusion goes from t=1.0 → t=0.0, so step 0 is t≈1 and last step
        # is t≈0.  Song position should be the reverse: step 0 is song start.
        # However, the temporal schedule maps to *song content position*, not
        # to diffusion timestep.  During diffusion, ALL positions are refined
        # simultaneously — there's no per-frame temporal ordering.
        #
        # Instead, we map the diffusion step linearly across the schedule:
        # step 0 / total → 0.0, last step → 1.0.  This spreads the adapter
        # influence evenly across diffusion steps.
        if total_steps > 1:
            position = step_idx / (total_steps - 1)
        else:
            position = 0.5

        scales = schedule.get_effective_scales(position)
        _apply_merged_weights_temporal(self, scales)

    return _on_step


class AdvancedAdapterMixin:
    """Multi-adapter slot system with weight-space merging and per-group scaling.

    Expected host attributes:
    - model, device, dtype, quantization
    - _base_decoder, use_lora, lora_loaded
    - _adapter_slots, _next_slot_id, _merged_dirty, lora_group_scales
    - _temporal_schedule (Optional[TemporalAdapterSchedule])
    """

    _extract_adapter_delta = _extract_adapter_delta
    _apply_merged_weights_advanced = _apply_merged_weights
    _apply_merged_weights_with_groups = _apply_merged_weights_with_groups
    _apply_merged_weights_temporal = _apply_merged_weights_temporal

    load_lora_slot = load_lora_slot
    unload_lora_slot = unload_lora_slot
    set_use_lora_advanced = set_use_lora_advanced
    set_lora_slot_scale = set_lora_slot_scale
    set_lora_group_scales = set_lora_group_scales
    set_slot_group_scales = set_slot_group_scales
    set_slot_layer_scales = set_slot_layer_scales
    set_slot_layer_scale = set_slot_layer_scale
    set_temporal_schedule = set_temporal_schedule
    build_temporal_step_callback = build_temporal_step_callback
    get_advanced_lora_status = get_advanced_lora_status
