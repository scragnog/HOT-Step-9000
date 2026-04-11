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


def _log_vram(label: str) -> None:
    """Log current CUDA VRAM usage with a contextual label."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    logger.info(f"[VRAM] {label}: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB")


def _dequantize_decoder_nf4(model) -> int:
    """Dequantize all NF4Tensor weights in the decoder back to bfloat16.

    This must be called before adapter operations (backup, PEFT injection, merge)
    because NF4Tensor does not support in-place operations, state_dict round-trips,
    or parameter assignment that the adapter pipeline requires.

    Uses ``weight.data.float()`` which goes through NF4Tensor's
    ``__torch_dispatch__`` to properly dequantize, then casts to bfloat16.
    This is more reliable than ``.dequantize()`` which is a classmethod in
    some torchao versions (requires ``(value, nf4)`` positional args).

    Returns:
        Number of parameters dequantized.
    """
    import torch.nn as nn
    _log_vram("before dequantize")
    logger.info("[NF4 compat] Starting decoder dequantization...")
    count = 0
    errors = 0
    for _name, module in model.decoder.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        w = module.weight
        w_type = type(w.data).__name__

        # Skip already-dequantized weights (plain Tensor / Parameter)
        if w_type in ("Tensor", "Parameter"):
            continue

        try:
            # float() goes through NF4Tensor.__torch_dispatch__ → proper dequant
            deq = w.data.float().to(torch.bfloat16).detach()
            module.weight = nn.Parameter(deq, requires_grad=False)
            count += 1
        except Exception as e:
            logger.error(
                f"[NF4 compat] Cannot dequantize {_name} (type={w_type}): {e}"
            )
            errors += 1

    logger.info(
        f"[NF4 compat] Dequantized {count} linear layers for adapter operations"
        + (f" ({errors} errors)" if errors else "")
    )
    _log_vram("after dequantize")
    return count


def _requantize_decoder_nf4(model, skip_parts=("tokenizer", "detokenizer")) -> int:
    """Re-apply NF4 quantization to the decoder's linear layers after merge.

    Called after adapter weight merging to restore VRAM savings.

    Args:
        model: The model containing a .decoder attribute.
        skip_parts: Module name segments to exclude from quantization.

    Returns:
        Number of parameters re-quantized.
    """
    import torch.nn as nn
    try:
        from torchao.dtypes import to_nf4
        from torchao.dtypes.nf4tensor import NF4Tensor
    except ImportError:
        logger.warning("[NF4 compat] torchao not available — cannot re-quantize")
        return 0

    count = 0
    for name, module in model.decoder.named_modules():
        if isinstance(module, nn.Linear):
            skip = any(part in name.split(".") for part in skip_parts)
            # Check if the weight is already NF4 by inspecting the actual tensor type.
            # NOTE: hasattr(t, 'dequantize') is True for ALL torch.Tensor objects
            # in modern PyTorch — it's a built-in method on the Tensor class.
            # We must check the concrete type instead.
            is_already_nf4 = isinstance(module.weight.data, NF4Tensor)
            if not skip and not is_already_nf4:
                module.weight = nn.Parameter(
                    to_nf4(module.weight.data), requires_grad=False
                )
                count += 1
    if count:
        logger.info(f"[NF4 compat] Re-quantized {count} linear layers after merge")
        _log_vram("after NF4 re-quantize")
    else:
        logger.debug("[NF4 compat] Re-quantize: 0 layers needed conversion (all already NF4)")
    return count


def _determine_group(module_name: str) -> str:
    """Determine which module group a named module belongs to.

    Key naming in the decoder's LinearTransformerBlock:
      - .cross_attn.     → cross-attention (checked first, contains 'attn')
      - .attn.           → self / joint attention
      - .ff.             → feed-forward / MLP
      - condition_embed*  → text conditioning embedding
    """
    if "cross_attn" in module_name:
        return "cross_attn"
    elif ".attn." in module_name or ".attn_" in module_name:
        return "self_attn"
    elif ".ff." in module_name or ".ff_" in module_name:
        return "mlp"
    elif "condition_embed" in module_name:
        return "cond_embed"
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
    _bare_peft_tmp_dir = None
    if os.path.isdir(lora_path):
        if os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            is_peft = True
        st_files = sorted(glob.glob(os.path.join(lora_path, "*.safetensors")))
        if not is_peft and st_files:
            # No adapter_config.json — try bare PEFT detection before assuming LoKr
            from acestep.core.generation.handler.lora.lifecycle import _try_prepare_bare_peft_safetensors
            for sf in st_files:
                tmp = _try_prepare_bare_peft_safetensors(sf)
                if tmp:
                    _bare_peft_tmp_dir = tmp
                    lora_path = tmp
                    is_peft = True
                    break
            if not is_peft:
                lokr_first = [f for f in st_files if "lokr" in os.path.basename(f).lower()]
                lokr_weights_path = lokr_first[0] if lokr_first else st_files[0]
        elif not is_peft and not st_files:
            pass  # will fall through to error
        elif is_peft and st_files:
            # Also set lokr_weights_path for trigger word extraction below
            lokr_first = [f for f in st_files if "lokr" in os.path.basename(f).lower()]
            lokr_weights_path = lokr_first[0] if lokr_first else st_files[0]
    elif lora_path.endswith(".safetensors"):
        # User selected a .safetensors file directly — check whether the
        # parent directory is a PEFT adapter dir before assuming LoKr.
        parent = os.path.dirname(lora_path)
        if os.path.exists(os.path.join(parent, "adapter_config.json")):
            lora_path = parent
            is_peft = True
        else:
            # Try bare PEFT detection before assuming LoKr
            from acestep.core.generation.handler.lora.lifecycle import _try_prepare_bare_peft_safetensors
            tmp = _try_prepare_bare_peft_safetensors(lora_path)
            if tmp:
                _bare_peft_tmp_dir = tmp
                lokr_weights_path = lora_path  # preserve for trigger word extraction
                lora_path = tmp
                is_peft = True
            else:
                lokr_weights_path = lora_path

    # Reset dynamo state — nano-vllm's global config changes
    # (capture_scalar_outputs, @torch.compile decorators) contaminate
    # the compilation context, causing "Offset increment outside graph
    # capture" when we modify decoder weights.
    torch._dynamo.reset()
    self.model.decoder.load_state_dict(self._base_decoder, strict=False, assign=True)
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

        # Move model to CPU for lycoris injection
        original_device = self.device
        self.model.decoder = self.model.decoder.cpu()

        from acestep.core.generation.handler.lora.lifecycle import _load_lokr_adapter
        lycoris_net = _load_lokr_adapter(self.model.decoder, lokr_weights_path)

        # === Compute deltas directly from LoKR factor matrices ===
        # We intentionally bypass lycoris_net.merge_to() because LyCORIS has
        # a double-scaling bug: get_diff_weight() calls get_weight() which
        # already applies self.scale via make_kron(), then multiplies by
        # self.scale again.  When scale=1.0 (alpha==dim) this is invisible,
        # but for alpha!=dim the delta is scaled by scale² instead of scale.
        #
        # Instead we compute the delta using the same formula as LyCORIS's
        # training forward() path: get_weight(shape) * scalar.
        # This matches what the model actually saw during training.

        # Build a map from decoder parameter data_ptr → state dict key
        param_to_key = {}
        for key, param in self.model.decoder.named_parameters():
            param_to_key[param.data_ptr()] = key

        adapted_sd = {}  # will hold only the delta keys for LoKR
        lokr_delta_keys = set()
        for lora_mod in lycoris_net.loras:
            org_module = lora_mod.org_module[0]
            weight_ptr = org_module.weight.data_ptr()
            sd_key = param_to_key.get(weight_ptr)
            if sd_key is None:
                continue
            # get_weight includes self.scale once via make_kron — correct
            # scalar is 1.0 for standard adapters (non-use_scalar)
            diff = lora_mod.get_weight(org_module.weight.shape)
            scalar_val = lora_mod.scalar
            if scalar_val is not None:
                diff = diff.float() * scalar_val.float()
            else:
                diff = diff.float()
            if diff.abs().max().item() > 1e-8:
                adapted_sd[sd_key] = diff.detach().cpu()
                lokr_delta_keys.add(sd_key)

        logger.info(
            f"LoKr direct delta: {len(lokr_delta_keys)} keys computed from "
            f"{len(lycoris_net.loras)} modules (scale={getattr(lycoris_net.loras[0], 'scale', '?') if lycoris_net.loras else 'N/A'})"
        )

        # === Cleanup: restore decoder and remove LyCORIS hooks/wrappers ===
        try:
            lycoris_net.restore()
        except Exception:
            pass

        # Remove forward wrappers left by apply_to().  LyCORIS restore() only
        # un-bakes merge_to() weight changes but does NOT remove the forward
        # monkey-patches.  Clean up via the _lycoris_wrappers list if present,
        # otherwise fall back to _lycoris_original_forward.
        cleaned = 0
        for _name, mod in self.model.decoder.named_modules():
            wrappers = getattr(mod, '_lycoris_wrappers', None)
            if wrappers:
                orig_fwd = getattr(mod, '_lycoris_original_forward', None)
                if orig_fwd is not None:
                    mod.forward = orig_fwd
                mod.__dict__.pop('_lycoris_wrappers', None)
                mod.__dict__.pop('_lycoris_original_forward', None)
                cleaned += 1

        if cleaned:
            logger.info(f"Cleaned {cleaned} LyCORIS forward wrappers from decoder")

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

    # Compute delta — for LoKR, adapted_sd already contains raw deltas
    # (computed directly from factor matrices).  For PEFT, subtract base.
    delta = {}
    if adapter_type == "lycoris_lokr":
        delta = {k: v.float() for k, v in adapted_sd.items()}
    else:
        for k in self._base_decoder:
            if k in adapted_sd:
                diff = adapted_sd[k].float() - self._base_decoder[k].float()
                if diff.abs().max().item() > 1e-8:
                    delta[k] = diff
    del adapted_sd

    # Restore decoder to base state
    self.model.decoder.load_state_dict(self._base_decoder, strict=False, assign=True)
    self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
    self.model.decoder.eval()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    delta_mb = sum(v.numel() * 4 for v in delta.values()) / (1024**2)
    logger.info(f"Extracted adapter delta: {len(delta)} keys, {delta_mb:.1f}MB (fp32 on CPU)")

    # Clean up temporary PEFT directory created for bare safetensors
    if _bare_peft_tmp_dir and os.path.isdir(_bare_peft_tmp_dir):
        try:
            import shutil
            shutil.rmtree(_bare_peft_tmp_dir, ignore_errors=True)
        except Exception:
            pass

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
        self.model.decoder.load_state_dict(self._base_decoder, strict=False, assign=True)
        self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
        self.model.decoder.eval()
        # Re-quantize to NF4 if the model was originally NF4-quantized
        if getattr(self, 'quantization', None) == 'nf4':
            _requantize_decoder_nf4(self.model)
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
    self.model.decoder.load_state_dict(merged, strict=False, assign=True)
    self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
    self.model.decoder.eval()

    # Re-quantize to NF4 if the model was originally NF4-quantized
    if getattr(self, 'quantization', None) == 'nf4':
        _requantize_decoder_nf4(self.model)

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
    _log_vram("after merge + cleanup")

    elapsed = time.time() - t0
    slot_desc = ", ".join(
        f"slot{sid}={s['name']}@{s['scale']:.2f}" for sid, s in active_slots.items()
    )
    logger.info(f"Merged {len(active_slots)} adapter(s) in {elapsed:.1f}s: {slot_desc}")
    self._merged_dirty = False

    # Post-merge safety: ensure no LyCORIS hooks remain and all params are on device
    _verify_decoder_ready(self, label="post-merge")


def _verify_decoder_ready(self, *, label: str = "") -> dict:
    """Verify decoder state is clean and GPU-resident. Fix any issues found.

    Acts as both a diagnostic probe (logs exactly what's wrong) and a
    safety net (fixes whatever it finds) so generation never hits a device
    mismatch RuntimeError.

    Returns a dict with diagnostic counts for test introspection.
    """
    decoder = getattr(getattr(self, "model", None), "decoder", None)
    if decoder is None:
        return {}

    target_device = getattr(self, "device", "cuda")
    target_dtype = getattr(self, "dtype", torch.bfloat16)
    prefix = f"[_verify_decoder_ready/{label}]" if label else "[_verify_decoder_ready]"

    # -- 1) Detect & remove stale LyCORIS forward wrappers --
    lycoris_hooks_found = 0
    lycoris_forward_replaced = 0
    for name, mod in decoder.named_modules():
        # Check for _lycoris_wrappers attribute (set by LyCORIS apply_to)
        wrappers = getattr(mod, "_lycoris_wrappers", None)
        if wrappers:
            orig_fwd = getattr(mod, "_lycoris_original_forward", None)
            if orig_fwd is not None:
                mod.forward = orig_fwd
                lycoris_forward_replaced += 1
            mod.__dict__.pop("_lycoris_wrappers", None)
            mod.__dict__.pop("_lycoris_original_forward", None)
            lycoris_hooks_found += 1

        # Also check if the forward method itself is from a LyCORIS module
        # (safety net in case _lycoris_wrappers was somehow cleared but
        # forward was not restored)
        fwd = getattr(mod, "forward", None)
        if fwd is not None:
            fwd_self = getattr(fwd, "__self__", None)
            if fwd_self is not None and fwd_self is not mod:
                fwd_class = type(fwd_self).__name__
                if "Module" in fwd_class and fwd_class != type(mod).__name__:
                    # This module's forward is bound to a DIFFERENT object
                    # (likely a LokrModule/LoConModule/etc.)
                    orig_fwd = getattr(mod, "_lycoris_original_forward", None)
                    if orig_fwd is None:
                        # Last resort: use the class's original forward
                        orig_fwd = type(mod).forward
                    if orig_fwd is not None:
                        mod.forward = orig_fwd if not isinstance(orig_fwd, type) else orig_fwd.__get__(mod, type(mod))
                        lycoris_forward_replaced += 1
                    logger.warning(
                        f"{prefix} Module '{name}' had forward bound to "
                        f"{fwd_class} — restored to original"
                    )

    if lycoris_hooks_found > 0:
        logger.warning(
            f"{prefix} Found and removed {lycoris_hooks_found} stale LyCORIS "
            f"wrapper(s), restored {lycoris_forward_replaced} forward method(s)"
        )

    # -- 2) Find parameters on wrong device and move them --
    cpu_params = 0
    total_params = 0
    sample_cpu_names = []
    for pname, param in decoder.named_parameters():
        total_params += 1
        if param.device.type != str(target_device).split(":")[0]:
            cpu_params += 1
            if len(sample_cpu_names) < 5:
                sample_cpu_names.append(f"{pname}({param.device})")

    if cpu_params > 0:
        logger.warning(
            f"{prefix} {cpu_params}/{total_params} parameters on wrong device! "
            f"Samples: {sample_cpu_names}. Moving decoder to {target_device}."
        )
        decoder.to(target_device)
        decoder.to(target_dtype)
        decoder.eval()

        # Verify the move worked
        still_bad = sum(
            1 for _, p in decoder.named_parameters()
            if p.device.type != str(target_device).split(":")[0]
        )
        if still_bad > 0:
            logger.error(
                f"{prefix} FAILED to move {still_bad} parameters to {target_device} "
                f"— generation will likely fail!"
            )
        else:
            logger.info(f"{prefix} Successfully moved all parameters to {target_device}")

    # -- 3) Also check _lycoris_net lingering on decoder --
    if hasattr(decoder, "_lycoris_net") and decoder._lycoris_net is not None:
        logger.warning(f"{prefix} Stale _lycoris_net found on decoder — removing")
        try:
            decoder._lycoris_net.restore()
        except Exception:
            pass
        decoder._lycoris_net = None

    if lycoris_hooks_found == 0 and cpu_params == 0:
        logger.debug(f"{prefix} Decoder OK ({total_params} params on {target_device})")

    return {
        "lycoris_hooks_found": lycoris_hooks_found,
        "lycoris_forward_replaced": lycoris_forward_replaced,
        "cpu_params": cpu_params,
        "total_params": total_params,
    }


def _apply_merged_weights_with_groups(self) -> None:
    """Like _apply_merged_weights but applies per-group scaling."""
    if self._base_decoder is None:
        return

    active_slots = {
        sid: s for sid, s in self._adapter_slots.items()
        if s["scale"] > 0 and self.use_lora
    }

    if not active_slots:
        self.model.decoder.load_state_dict(self._base_decoder, strict=False, assign=True)
        self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
        self.model.decoder.eval()
        # Re-quantize to NF4 if the model was originally NF4-quantized
        if getattr(self, 'quantization', None) == 'nf4':
            _requantize_decoder_nf4(self.model)
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
                    gs = s.get("group_scales", {})
                    if group:
                        g_scale = gs.get(group, 1.0)
                    else:
                        # Unclassified keys (norms, final_layer, etc.)
                        # use the average of all group scales so they respect
                        # the user's intent when groups are zeroed out.
                        vals = [gs.get("self_attn", 1.0), gs.get("cross_attn", 1.0), gs.get("mlp", 1.0), gs.get("cond_embed", 1.0)]
                        g_scale = sum(vals) / len(vals)
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

    self.model.decoder.load_state_dict(merged, strict=False, assign=True)
    self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
    self.model.decoder.eval()

    # Re-quantize to NF4 if the model was originally NF4-quantized
    if getattr(self, 'quantization', None) == 'nf4':
        _requantize_decoder_nf4(self.model)

    del merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log_vram("after group-merge + cleanup")

    elapsed = time.time() - t0
    logger.info(f"Merged {len(active_slots)} adapter(s) with group scales in {elapsed:.1f}s")
    self._merged_dirty = False

    # Post-merge safety (group-scaled path)
    _verify_decoder_ready(self, label="post-group-merge")


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
        logger.warning(
            f"⚠️ Loading LoRA on quantized model ({self.quantization}) — "
            "this may fail or produce unexpected results. INT8 usually works, INT4/NF4 are risky."
        )

    if not lora_path or not lora_path.strip():
        return "❌ Please provide a LoRA path."
    lora_path = lora_path.strip()

    if not os.path.exists(lora_path):
        return f"❌ LoRA path not found: {lora_path}"

    if len(self._adapter_slots) >= MAX_ADAPTER_SLOTS:
        return f"❌ Maximum {MAX_ADAPTER_SLOTS} adapter slots. Please unload one first."

    try:
        # If NF4 quantized, dequantize first so backup and adapter ops work
        _needs_nf4_requant = False
        if getattr(self, 'quantization', None) == 'nf4':
            deq_count = _dequantize_decoder_nf4(self.model)
            _needs_nf4_requant = deq_count > 0

        # Backup base decoder on first load
        if self._base_decoder is None:
            _log_vram("before base decoder backup")
            logger.info("Backing up base decoder state_dict to CPU")
            backup = {}
            for k, v in self.model.decoder.state_dict().items():
                # After _dequantize_decoder_nf4, weights should be plain bf16.
                # Use float() as a safe universal path that handles any
                # remaining quantized tensors via __torch_dispatch__.
                try:
                    backup[k] = v.detach().cpu().clone()
                except Exception:
                    backup[k] = v.float().to(torch.bfloat16).detach().cpu().clone()
            self._base_decoder = backup
            backup_mb = sum(v.numel() * v.element_size() for v in self._base_decoder.values()) / (1024**2)
            logger.info(f"Base decoder backed up ({backup_mb:.1f}MB)")
            _log_vram("after base decoder backup")

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
            "group_scales": {"self_attn": 1.0, "cross_attn": 1.0, "mlp": 1.0, "cond_embed": 1.0},
            "layer_scales": {},  # empty = all layers at 1.0
        }

        self.use_lora = True
        self.lora_loaded = True
        self._merged_dirty = True

        # Extract trigger word metadata from safetensors header and store per-slot
        safetensors_file = result.get("safetensors_file")
        if safetensors_file:
            from acestep.core.generation.handler.lora.lifecycle import _read_trigger_word_from_safetensors
            tw, tp = _read_trigger_word_from_safetensors(safetensors_file)
            if tw:
                self._adapter_slots[slot]["trigger_word"] = tw
                self._adapter_slots[slot]["tag_position"] = tp or "prepend"
                # Keep legacy single attr for basic-mode compat
                self._adapter_trigger_word = tw
                self._adapter_tag_position = tp or "prepend"
                logger.info(f"Adapter trigger word: '{tw}' (position: {tp or 'prepend'})")
            else:
                self._adapter_slots[slot]["trigger_word"] = ""
                self._adapter_slots[slot]["tag_position"] = ""
                self._adapter_trigger_word = ""
                self._adapter_tag_position = ""

        _log_vram("before merge")
        _apply_merged_weights(self)

        # Re-quantize merged weights back to NF4 for VRAM savings
        if _needs_nf4_requant:
            _requantize_decoder_nf4(self.model)

        delta_keys = len(result["delta"])
        type_label = "LoRA" if result["type"] == "peft_lora" else "LoKr"
        _log_vram("adapter load complete")
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
            _log_vram("before unload all")
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
            _log_vram("after unload all")
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
        scale: Scale value (0.0–4.0)
        slot: Specific slot. None = all slots.
    """
    if not self._adapter_slots:
        return "⚠️ No adapters loaded"

    scale = max(0.0, min(4.0, scale))

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
    cond_embed_scale: float = 1.0,
) -> str:
    """Set per-module-group global scales applied to all slots."""
    scales = {
        "self_attn": max(0.0, min(4.0, self_attn_scale)),
        "cross_attn": max(0.0, min(4.0, cross_attn_scale)),
        "mlp": max(0.0, min(4.0, mlp_scale)),
        "cond_embed": max(0.0, min(4.0, cond_embed_scale)),
    }
    self.lora_group_scales = scales

    for s in self._adapter_slots.values():
        s["group_scales"] = dict(scales)

    if self._adapter_slots and self.use_lora:
        self._merged_dirty = True
        _apply_merged_weights_with_groups(self)

    sa, ca, ml, ce = scales["self_attn"], scales["cross_attn"], scales["mlp"], scales["cond_embed"]
    return f"✅ Group scales (all slots): SA={sa:.0%} CA={ca:.0%} MLP={ml:.0%} CE={ce:.0%}"


def set_slot_group_scales(
    self, slot: int, self_attn_scale: float = 1.0,
    cross_attn_scale: float = 1.0, mlp_scale: float = 1.0,
    cond_embed_scale: float = 1.0,
) -> str:
    """Set per-group LoRA scales for a specific adapter slot."""
    if slot not in self._adapter_slots:
        return f"❌ Slot {slot} not found. Active slots: {list(self._adapter_slots.keys())}"

    scales = {
        "self_attn": max(0.0, min(4.0, self_attn_scale)),
        "cross_attn": max(0.0, min(4.0, cross_attn_scale)),
        "mlp": max(0.0, min(4.0, mlp_scale)),
        "cond_embed": max(0.0, min(4.0, cond_embed_scale)),
    }
    self._adapter_slots[slot]["group_scales"] = scales

    if self.use_lora:
        self._merged_dirty = True
        _apply_merged_weights_with_groups(self)

    name = self._adapter_slots[slot]["name"]
    sa, ca, ml, ce = scales["self_attn"], scales["cross_attn"], scales["mlp"], scales["cond_embed"]
    return f"✅ Slot {slot} ({name}) group scales: SA={sa:.0%} CA={ca:.0%} MLP={ml:.0%} CE={ce:.0%}"


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
            "group_scales": s.get("group_scales", {"self_attn": 1.0, "cross_attn": 1.0, "mlp": 1.0, "cond_embed": 1.0}),
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
        self.model.decoder.load_state_dict(self._base_decoder, strict=False, assign=True)
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
                    gs = s.get("group_scales", {})
                    if group:
                        g_scale = gs.get(group, 1.0)
                    else:
                        vals = [gs.get("self_attn", 1.0), gs.get("cross_attn", 1.0), gs.get("mlp", 1.0), gs.get("cond_embed", 1.0)]
                        g_scale = sum(vals) / len(vals)
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
    self.model.decoder.load_state_dict(merged, strict=False, assign=True)
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
