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
    This wraps each DiT layer's forward AND self_attn.forward to intercept
    self-attention and replace output with identity for PAG-marked batch items.

    We can't use register_forward_hook because self_attn receives hidden_states
    as a keyword arg → hook's `inp` tuple is empty → identity replacement fails.
    Direct method wrapping gives full access to args and kwargs.

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

        # -- Step 1: Wrap self_attn.forward to intercept attention output --
        original_self_attn_forward = self_attn.forward

        def _make_pag_self_attn_forward(orig_sa_fwd, layer_ref):
            def pag_self_attn_forward(*args, **kwargs):
                result = orig_sa_fwd(*args, **kwargs)

                pag_mask = getattr(layer_ref, "_pag_active_mask", None)
                if pag_mask is not None and pag_mask.any():
                    # Get the input hidden_states (positional or keyword)
                    hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                    if hs is not None:
                        attn_out = result[0]
                        # Replace PAG items' attention output with identity
                        attn_out[pag_mask] = hs[pag_mask]

                return result
            return pag_self_attn_forward

        self_attn.forward = _make_pag_self_attn_forward(original_self_attn_forward, layer)

        # -- Step 2: Wrap layer.forward to set/clear PAG mask --
        original_layer_forward = layer.forward

        def _make_pag_layer_forward(orig_fwd, layer_ref):
            def pag_layer_forward(*args, **kwargs):
                pag_mask = kwargs.pop("pag_identity_mask", None)
                layer_ref._pag_active_mask = pag_mask
                try:
                    return orig_fwd(*args, **kwargs)
                finally:
                    layer_ref._pag_active_mask = None
            return pag_layer_forward

        layer.forward = _make_pag_layer_forward(original_layer_forward, layer)
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

    # Also monkeypatch prepare_condition — the cached checkpoint model
    # does NOT have precomputed_lm_hints_25Hz support, so audio codes
    # from LM thinking mode would be silently ignored without this.
    patched_prepare_condition = BaseModel.prepare_condition
    model.prepare_condition = types.MethodType(patched_prepare_condition, model)
    logger.info(
        f"[monkeypatch] Replaced prepare_condition on {class_name} "
        f"(added precomputed_lm_hints_25Hz / audio code conditioning support)"
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

