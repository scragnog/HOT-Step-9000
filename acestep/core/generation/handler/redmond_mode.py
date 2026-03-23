"""Redmond Mode — DPO-refined base quality adapter.

Merges the AceStep_Refine_Redmond adapter directly into the DiT decoder
weights *below* the adapter slot system.  The slot system's ``_base_decoder``
backup will contain Redmond-enhanced weights, so all artist adapter deltas
are computed against the improved base.

State stored on the handler:
    _raw_base_decoder   — truly original weights (CPU, before any refinement)
    _redmond_delta      — Redmond adapter delta (CPU, fp32)
    _redmond_scale      — current scale (float)
    _redmond_enabled    — whether active (bool)
    _redmond_adapter_path — path to adapter directory
"""

from __future__ import annotations

import gc
import os
import time
from typing import Any, Dict, Optional

import torch
from loguru import logger


def _ensure_peft_filename(adapter_dir: str) -> None:
    """Rename the safetensors weights file to adapter_model.safetensors if needed.

    HuggingFace repos sometimes use custom filenames (e.g.
    ``AceStep_Refine_Redmond_standard.safetensors``) but PEFT's
    ``from_pretrained`` expects ``adapter_model.safetensors``.
    """
    expected = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.isfile(expected):
        return  # Already correct

    # Find any .safetensors file in the directory
    for fname in os.listdir(adapter_dir):
        if fname.endswith(".safetensors"):
            src = os.path.join(adapter_dir, fname)
            os.rename(src, expected)
            logger.info(f"[Redmond] Renamed {fname} → adapter_model.safetensors")
            return


def _compute_redmond_delta(handler: Any, adapter_path: str) -> Dict[str, torch.Tensor]:
    """Load PEFT adapter, merge into decoder, compute delta vs raw base.

    Returns dict of weight key → fp32 delta tensor (CPU).
    """
    from peft import PeftModel

    # Snapshot current decoder state (the raw base)
    raw_sd = {k: v.detach().cpu().clone() for k, v in handler.model.decoder.state_dict().items()}

    # Reset dynamo state to avoid "Offset increment outside graph capture"
    torch._dynamo.reset()

    # Move to CPU for PEFT loading (avoids CUDA graph issues)
    original_device = handler.device
    handler.model.decoder = handler.model.decoder.cpu()
    handler.model.decoder = PeftModel.from_pretrained(
        handler.model.decoder, adapter_path, is_trainable=False,
    )
    handler.model.decoder = handler.model.decoder.to(original_device).to(handler.dtype)
    handler.model.decoder.eval()
    handler.model.decoder = handler.model.decoder.merge_and_unload()

    # Capture adapted state
    adapted_sd = {k: v.detach().cpu().clone() for k, v in handler.model.decoder.state_dict().items()}

    # Compute delta = adapted - raw (CPU, fp32, only changed keys)
    delta = {}
    for k in raw_sd:
        if k in adapted_sd:
            diff = adapted_sd[k].float() - raw_sd[k].float()
            if diff.abs().max().item() > 1e-8:
                delta[k] = diff
    del adapted_sd

    # Restore decoder to raw base state
    handler.model.decoder.load_state_dict(raw_sd, strict=False)
    handler.model.decoder = handler.model.decoder.to(original_device).to(handler.dtype)
    handler.model.decoder.eval()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    delta_mb = sum(v.numel() * 4 for v in delta.values()) / (1024**2)
    logger.info(f"[Redmond] Extracted delta: {len(delta)} keys, {delta_mb:.1f}MB (fp32 on CPU)")

    return delta


def _apply_redmond_to_decoder(handler: Any) -> None:
    """Recompute decoder weights: raw_base + (redmond_scale × redmond_delta).

    Also refreshes _base_decoder so the adapter slot system sees the
    Redmond-enhanced weights as its baseline.
    """
    raw = handler._raw_base_decoder
    delta = handler._redmond_delta
    scale = handler._redmond_scale if handler._redmond_enabled else 0.0

    t0 = time.time()
    merged = {}
    delta_keys = set(delta.keys())

    for k in raw:
        base_val = raw[k]
        if k in delta_keys and scale > 0:
            merged[k] = (base_val.float() + scale * delta[k]).to(dtype=base_val.dtype)
        else:
            merged[k] = base_val

    torch._dynamo.reset()
    handler.model.decoder.load_state_dict(merged, strict=False)
    handler.model.decoder = handler.model.decoder.to(handler.device).to(handler.dtype)
    handler.model.decoder.eval()

    # Update _base_decoder so adapter slot system uses Redmond as its baseline
    handler._base_decoder = {k: v.detach().cpu().clone() for k, v in merged.items()}

    del merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    status = f"scale={scale:.2f}" if handler._redmond_enabled else "disabled"
    logger.info(f"[Redmond] Decoder rebuilt in {elapsed:.1f}s ({status})")


# ── Public API ──────────────────────────────────────────────────────────


def apply_redmond_at_startup(handler: Any, adapter_path: str, scale: float = 0.7) -> str:
    """Merge Redmond adapter into the DiT decoder at startup.

    Called once after ``handler.initialize_service()`` succeeds.
    Stores raw base + delta so runtime toggle is possible.

    Args:
        handler: AceStepHandler instance with loaded model.
        adapter_path: Path to the PEFT adapter directory.
        scale: Initial merge scale (0.0–2.0).

    Returns:
        Status message string.
    """
    if handler.model is None:
        return "❌ Model not initialized"
    if not os.path.isdir(adapter_path):
        return f"❌ Redmond adapter not found: {adapter_path}"

    t0 = time.time()
    _ensure_peft_filename(adapter_path)
    logger.info(f"[Redmond] Merging base refinement adapter at scale {scale:.2f}")
    logger.info(f"[Redmond] Adapter path: {adapter_path}")

    # Store truly original decoder weights
    handler._raw_base_decoder = {
        k: v.detach().cpu().clone() for k, v in handler.model.decoder.state_dict().items()
    }

    # Compute delta
    handler._redmond_delta = _compute_redmond_delta(handler, adapter_path)
    handler._redmond_scale = max(0.0, min(2.0, scale))
    handler._redmond_enabled = True
    handler._redmond_adapter_path = adapter_path

    # Apply: decoder = raw + scale × delta
    _apply_redmond_to_decoder(handler)

    elapsed = time.time() - t0
    key_count = len(handler._redmond_delta)
    logger.info(f"[Redmond] Startup merge complete in {elapsed:.1f}s ({key_count} keys at scale {scale:.2f})")
    return f"✅ Redmond Mode active (scale {scale:.2f}, {elapsed:.1f}s)"


def toggle_redmond_mode(handler: Any, enabled: bool) -> str:
    """Enable or disable Redmond Mode at runtime.

    If enabling and the adapter hasn't been loaded yet, auto-downloads
    and merges it. Recomputes decoder weights and re-applies adapter slots.

    Args:
        handler: AceStepHandler instance.
        enabled: Whether to enable Redmond Mode.

    Returns:
        Status message string.
    """
    has_delta = hasattr(handler, "_redmond_delta") and handler._redmond_delta is not None
    has_raw = hasattr(handler, "_raw_base_decoder") and handler._raw_base_decoder is not None

    # If enabling but delta not loaded, try to load (auto-download if needed)
    if enabled and not has_delta:
        adapter_path = getattr(handler, "_redmond_adapter_path", "")

        # Try to find/download the adapter
        if not adapter_path or not os.path.isdir(adapter_path):
            # Construct default path and auto-download
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            checkpoint_dir = os.path.join(project_root, "checkpoints")
            adapter_path = os.path.join(checkpoint_dir, "redmond-refine", "standard")

            if not os.path.isdir(adapter_path):
                logger.info("[Redmond] Adapter not found, downloading...")
                try:
                    from huggingface_hub import snapshot_download
                    redmond_parent = os.path.join(checkpoint_dir, "redmond-refine")
                    os.makedirs(redmond_parent, exist_ok=True)
                    snapshot_download(
                        "artificialguybr/AceStep_Refine_Redmond",
                        allow_patterns="standard/*",
                        local_dir=redmond_parent,
                    )
                except Exception as exc:
                    return f"❌ Failed to download Redmond adapter: {exc}"

        if not os.path.isdir(adapter_path):
            return "❌ Redmond adapter not available"

        # Load and merge
        result = apply_redmond_at_startup(handler, adapter_path, getattr(handler, "_redmond_scale", 0.7))
        if result.startswith("❌"):
            return result
        return f"✅ Redmond Mode enabled (downloaded and merged)"

    if not has_delta:
        return "❌ Redmond adapter not loaded."
    if not has_raw:
        return "❌ Raw base decoder not available."

    prev = handler._redmond_enabled
    handler._redmond_enabled = enabled

    if prev == enabled:
        status = "enabled" if enabled else "disabled"
        return f"✅ Redmond Mode already {status}"

    # Recompute decoder weights
    _apply_redmond_to_decoder(handler)

    # Re-apply adapter slots if any are loaded
    if hasattr(handler, "_adapter_slots") and handler._adapter_slots and handler.use_lora:
        from acestep.core.generation.handler.lora.advanced_adapter_mixin import (
            _apply_merged_weights_with_groups,
        )
        handler._merged_dirty = True
        _apply_merged_weights_with_groups(handler)

    status = "enabled" if enabled else "disabled"
    return f"✅ Redmond Mode {status}"


def set_redmond_scale(handler: Any, scale: float) -> str:
    """Change Redmond Mode scale at runtime.

    Args:
        handler: AceStepHandler instance.
        scale: New scale value (0.0–2.0).

    Returns:
        Status message string.
    """
    if not hasattr(handler, "_redmond_delta") or handler._redmond_delta is None:
        return "❌ Redmond adapter not loaded."
    if not hasattr(handler, "_raw_base_decoder") or handler._raw_base_decoder is None:
        return "❌ Raw base decoder not available."

    scale = max(0.0, min(2.0, scale))
    handler._redmond_scale = scale

    if not handler._redmond_enabled:
        return f"✅ Redmond scale set to {scale:.2f} (will apply when enabled)"

    # Recompute decoder weights
    _apply_redmond_to_decoder(handler)

    # Re-apply adapter slots if any are loaded
    if hasattr(handler, "_adapter_slots") and handler._adapter_slots and handler.use_lora:
        from acestep.core.generation.handler.lora.advanced_adapter_mixin import (
            _apply_merged_weights_with_groups,
        )
        handler._merged_dirty = True
        _apply_merged_weights_with_groups(handler)

    return f"✅ Redmond scale: {scale:.2f}"


def get_redmond_status(handler: Any) -> Dict[str, Any]:
    """Get current Redmond Mode state.

    Returns:
        Dict with keys: enabled, scale, available, adapter_path, delta_keys.
    """
    has_delta = hasattr(handler, "_redmond_delta") and handler._redmond_delta is not None
    return {
        "enabled": getattr(handler, "_redmond_enabled", False),
        "scale": getattr(handler, "_redmond_scale", 0.7),
        "available": has_delta,
        "adapter_path": getattr(handler, "_redmond_adapter_path", ""),
        "delta_keys": len(handler._redmond_delta) if has_delta else 0,
    }


def reset_redmond_state(handler: Any) -> None:
    """Clear all Redmond state.  Called during model switch / reinit."""
    handler._raw_base_decoder = None
    handler._redmond_delta = None
    handler._redmond_scale = 0.7
    handler._redmond_enabled = False
    # Keep _redmond_adapter_path so we can re-apply after model switch
    logger.info("[Redmond] State reset (delta cleared)")
