"""
Guidance registry for ACE-Step flow matching diffusion.

IMPORTANT DESIGN NOTE:
The ACE-Step model was trained and designed to work exclusively with APG
(Adaptive Perpendicular Guidance) as implemented in apg_guidance.py.
The vanilla upstream code ONLY uses apg_forward — there is no "plain CFG".

All guidance modes therefore MUST pass through the full APG pipeline
(momentum smoothing + norm thresholding + perpendicular projection) to avoid
producing unbalanced audio (loud vocals, muffled instruments).

Modes differ by applying post-processing or scale adjustments to the
APG-guided output, NOT by replacing the core formula.

Guidance interface:
    guidance_fn(pred_cond, pred_uncond, guidance_scale, **ctx) -> vt_guided
"""

import math
import torch


# ---------------------------------------------------------------------------
# Core APG wrapper — all modes route through here
# ---------------------------------------------------------------------------

def _apg_core(pred_cond, pred_uncond, guidance_scale, **ctx):
    """Call apg_forward with proper momentum buffer handling.

    This is the ONLY function that actually computes guidance.
    All modes call this, then optionally post-process the result.
    """
    from acestep.models.base.apg_guidance import apg_forward
    momentum_buffer = ctx.get("momentum_buffer")
    if ctx.get("disable_momentum", False):
        momentum_buffer = None
    norm_threshold = ctx.get("norm_threshold", 2.5)
    erg_scale = ctx.get("erg_scale", 1.0)
    # ERG: scale the effective guidance to control prediction diversity
    effective_scale = guidance_scale * erg_scale if erg_scale != 1.0 else guidance_scale
    return apg_forward(
        pred_cond=pred_cond,
        pred_uncond=pred_uncond,
        guidance_scale=effective_scale,
        momentum_buffer=momentum_buffer,
        norm_threshold=norm_threshold,
        dims=[1],
    )


# ---------------------------------------------------------------------------
# Guidance functions
# ---------------------------------------------------------------------------



def cfg_pp(pred_cond, pred_uncond, guidance_scale, **ctx):
    """CFG++ — Reduced guidance for large steps.

    Uses APG with a step-scaled effective guidance to prevent
    over-correction when step size is large relative to sigma.
    """
    dt = ctx.get("dt")
    t_curr = ctx.get("sigma")

    if isinstance(dt, torch.Tensor): dt = dt.item()
    if isinstance(t_curr, torch.Tensor): t_curr = t_curr.item()

    if dt is not None and t_curr is not None and t_curr > 1e-6:
        step_scale = abs(dt) / t_curr
        effective_scale = 1.0 + (guidance_scale - 1.0) * step_scale
    else:
        effective_scale = guidance_scale

    return _apg_core(pred_cond, pred_uncond, effective_scale, **ctx)


def dynamic_cfg(pred_cond, pred_uncond, guidance_scale, **ctx):
    """Dynamic CFG — Cosine-decaying guidance schedule.

    Uses APG with full guidance early (establishing structure) and
    reduced guidance later (preserving fine detail).
    """
    step_idx = ctx.get("step_idx", 0)
    total_steps = ctx.get("total_steps", 1)
    power = 0.5

    progress = step_idx / max(total_steps - 1, 1)
    decay = math.cos(math.pi / 2 * progress) ** power
    effective_scale = 1.0 + (guidance_scale - 1.0) * decay

    return _apg_core(pred_cond, pred_uncond, effective_scale, **ctx)


def rescaled_cfg(pred_cond, pred_uncond, guidance_scale, **ctx):
    """Rescaled CFG — Std-matched post-processing.

    Runs APG at the requested scale, then rescales the output to match
    the conditional prediction's standard deviation, preventing
    over-saturation from high guidance. Blends rescaled and raw output.
    """
    phi = 0.95 if guidance_scale > 4.0 else 0.7

    guided = _apg_core(pred_cond, pred_uncond, guidance_scale, **ctx)

    # Post-process: match std of conditional prediction
    std_cond = pred_cond.std(dim=[1, 2], keepdim=True)
    std_guided = guided.std(dim=[1, 2], keepdim=True)
    factor = std_cond / (std_guided + 1e-5)
    rescaled = guided * factor

    # Blend rescaled and raw
    return phi * rescaled + (1 - phi) * guided


def apg_guidance(pred_cond, pred_uncond, guidance_scale, **ctx):
    """APG — Adaptive Perpendicular Guidance (native upstream default)."""
    return _apg_core(pred_cond, pred_uncond, guidance_scale, **ctx)


def adg_guidance(pred_cond, pred_uncond, guidance_scale, **ctx):
    """ADG — Angle-based Dynamic Guidance.

    Uses a completely different algorithm (angle-based) from
    the apg_guidance module.
    """
    from acestep.models.base.apg_guidance import adg_forward
    latents = ctx.get("latents")
    sigma = ctx.get("sigma")
    if latents is None or sigma is None:
        # Fallback to APG if missing required context
        return _apg_core(pred_cond, pred_uncond, guidance_scale, **ctx)
    return adg_forward(
        latents=latents,
        noise_pred_cond=pred_cond,
        noise_pred_uncond=pred_uncond,
        sigma=sigma,
        guidance_scale=guidance_scale,
    )


def pag_guidance(pred_cond, pred_uncond, guidance_scale, **ctx):
    """PAG — Perturbed Attention Guidance.

    True PAG per arXiv:2403.17377.  The main loop runs a triple-batch
    forward pass [cond, uncond, pag] and calls pag_combined_guidance()
    with all three predictions.  This function is the registry entry;
    it is NOT called directly during PAG generation.
    """
    # Fallback: if called without PAG predictions, just do APG
    return _apg_core(pred_cond, pred_uncond, guidance_scale, **ctx)


def pag_combined_guidance(pred_cond, pred_uncond, pred_pag,
                          guidance_scale, pag_scale, **ctx):
    """Combined APG + PAG guidance for triple-batch mode.

    Formula:  guided = apg_result + pag_scale * (pred_cond - pred_pag)

    The APG term handles conditional-vs-unconditional guidance.
    The PAG term adds structural coherence by guiding away from
    the perturbed-attention prediction.
    """
    # Standard APG guidance
    apg_result = _apg_core(pred_cond, pred_uncond, guidance_scale, **ctx)

    # PAG term: guide away from perturbed prediction.
    # Apply perpendicular projection to the PAG diff — same principle as APG:
    # the parallel component would amplify whatever's already dominant (vocals),
    # so we keep only the orthogonal component that adds new structural info.
    from acestep.models.base.apg_guidance import project
    pag_diff = pred_cond - pred_pag
    _parallel, pag_orthogonal = project(pag_diff, pred_cond, dims=[1])
    return apg_result + pag_scale * pag_orthogonal


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GUIDANCE_MODES = {
    "cfg_pp": cfg_pp,
    "dynamic_cfg": dynamic_cfg,
    "rescaled_cfg": rescaled_cfg,
    "apg": apg_guidance,
    "adg": adg_guidance,
    "pag": pag_guidance,
}

GUIDANCE_INFO = {
    "cfg_pp":       {"name": "CFG++",        "description": "Step-scaled guidance for few-step"},
    "dynamic_cfg":  {"name": "Dynamic CFG",  "description": "Cosine-decaying guidance schedule"},
    "rescaled_cfg": {"name": "Rescaled CFG", "description": "Std-matched to prevent saturation"},
    "apg":          {"name": "APG",          "description": "Adaptive perpendicular guidance"},
    "adg":          {"name": "ADG",          "description": "Angle-based dynamic guidance"},
    "pag":          {"name": "PAG",          "description": "Perturbed attention guidance"},
}

VALID_GUIDANCE = set(GUIDANCE_MODES.keys())


def get_guidance(name: str):
    """Get a guidance function by name."""
    name = name.lower()
    if name not in GUIDANCE_MODES:
        valid = ", ".join(sorted(VALID_GUIDANCE))
        raise ValueError(f"Unknown guidance mode '{name}'. Valid modes: {valid}")
    return GUIDANCE_MODES[name]
