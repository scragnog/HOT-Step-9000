"""
Runtime monkeypatch for generate_audio on checkpoint-loaded models.

Instead of modifying checkpoint files on disk (which breaks other apps),
this module replaces the generate_audio method on model instances AFTER
they are loaded via AutoModel.from_pretrained().

Our local model files (acestep/models/base/, acestep/models/sft/) already
contain the patched generate_audio with solver/guidance registry support.
Turbo models have their own distinct generate_audio and are not patched.

Additionally, we patch the decoder layer forward to support PAG (Perturbed
Attention Guidance) identity masks, since the checkpoint's layer code
doesn't have this feature.
"""

import types

import torch
from loguru import logger


def _patch_decoder_layer_for_pag(model) -> int:
    """Patch decoder layers to support pag_identity_mask in their forward.

    The checkpoint's AceStepDiTLayer.forward doesn't handle pag_identity_mask.
    This wraps each DiT layer's forward to intercept self-attention and replace
    the attention output with identity for PAG-marked batch items.

    Returns the number of layers patched.
    """
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return 0

    layers = getattr(decoder, "layers", None)
    if layers is None:
        return 0

    count = 0
    for layer in layers:
        # Only patch DiT layers that have self-attention (not encoder layers)
        self_attn = getattr(layer, "self_attn", None)
        self_attn_norm = getattr(layer, "self_attn_norm", None)
        if self_attn is None or self_attn_norm is None:
            continue

        # Check if already patched
        if getattr(layer, "_pag_patched", False):
            continue

        original_forward = layer.forward

        def _make_pag_forward(orig_fwd, layer_ref):
            """Create a PAG-aware wrapper for the layer forward."""
            def pag_forward(*args, **kwargs):
                pag_mask = kwargs.pop("pag_identity_mask", None)
                if pag_mask is None or not pag_mask.any():
                    return orig_fwd(*args, **kwargs)

                # PAG is active: we need to intercept self-attention output.
                # Strategy: hook self_attn to capture its output, then replace
                # PAG items with identity (the normalized input).
                captured = {}

                # Hook the self-attention module to capture input/output
                def _hook(module, inp, output):
                    # output is (attn_output, attn_weights)
                    attn_out = output[0]
                    # The input to self_attn is norm_hidden_states (after AdaLN)
                    norm_hs = inp[0] if len(inp) > 0 else None
                    if norm_hs is not None:
                        # Replace PAG items' attention output with identity
                        attn_out[pag_mask] = norm_hs[pag_mask]
                    captured["hooked"] = True
                    return (attn_out,) + output[1:]

                hook_handle = layer_ref.self_attn.register_forward_hook(_hook)
                try:
                    result = orig_fwd(*args, **kwargs)
                finally:
                    hook_handle.remove()

                return result
            return pag_forward

        layer.forward = _make_pag_forward(original_forward, layer)
        layer._pag_patched = True
        count += 1

    return count


def apply_generate_audio_monkeypatch(model) -> bool:
    """Replace generate_audio on *model* with our local patched version.

    Returns True if the monkeypatch was applied, False if skipped (e.g. turbo).
    """
    class_name = type(model).__name__
    module_name = type(model).__module__ or ""

    # Turbo models have a fundamentally different diffusion loop
    # (fixed timestep schedule, no solver/guidance registry). Skip them.
    # Use the config flag (authoritative) rather than module-name heuristics,
    # because merged SFT+Turbo models share turbo's model code but should
    # still receive the patched generate_audio for guidance/solver support.
    is_turbo = getattr(getattr(model, "config", None), "is_turbo", False)
    if is_turbo:
        logger.info(
            f"[monkeypatch] Skipping turbo model ({class_name}) — "
            "turbo has its own generate_audio"
        )
        return False

    # Import the patched generate_audio from our local base model code.
    # This version has: guidance_mode, get_solver(), get_guidance(), PAG, etc.
    from acestep.models.base.modeling_acestep_v15_base import (
        AceStepConditionGenerationModel as BaseModel,
    )

    patched_method = BaseModel.generate_audio

    # Bind the unbound method to the model instance
    model.generate_audio = types.MethodType(patched_method, model)

    logger.info(
        f"[monkeypatch] Replaced generate_audio on {class_name} "
        f"(from {module_name}) with local patched version"
    )

    # Also patch decoder layers for PAG identity mask support.
    # The checkpoint's layer code doesn't have pag_identity_mask handling,
    # so without this, the PAG identity mask would be silently ignored
    # and pred_pag would equal pred_cond (making PAG identical to APG).
    pag_count = _patch_decoder_layer_for_pag(model)
    if pag_count > 0:
        logger.info(
            f"[monkeypatch] Patched {pag_count} decoder layers for PAG identity mask support"
        )

    return True

