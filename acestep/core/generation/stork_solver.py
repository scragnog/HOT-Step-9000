"""
STORK Solvers for ACE-Step flow matching diffusion.

Implements STORK2 and STORK4 from the paper:
  "STORK: Faster Diffusion And Flow Matching Sampling By Resolving Both
   Stiffness And Structure-Dependence" (Tan et al., 2025, arXiv:2505.24210)

STORK uses Stabilized Runge-Kutta methods with Taylor expansion of the
velocity field to handle stiff ODEs in flow matching.  The key insight is
that the velocity derivatives are approximated from history (no extra model
calls), and cheap arithmetic sub-steps handle the stiffness.

Adaptive sub-stepping: if the Chebyshev polynomial evaluation produces
NaN/Inf (common in the high-noise regime t>0.7 where the velocity field
is rough), sub-steps are automatically halved until stable.  This means
the solver is always at least as good as Euler, and better where the ODE
is smooth enough for the polynomial to help.

Solver interface:
    solver_step(xt, vt, t_curr, t_prev, state, model_fn=None) -> (xt_next, state)

State dict keys used:
    velocity_history: list of (vt, dt) tuples
    step_index: int
    stork_substeps: int (default 10) -- overridable by caller
"""

import torch
import logging
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

Tensor = torch.Tensor
State = Dict[str, Any]
ModelFn = Callable[[Tensor, float], Tensor]

DEFAULT_SUBSTEPS = 10


def _euler_fallback(xt: Tensor, vt: Tensor, dt_val: float) -> Tensor:
    """Plain Euler step as safety fallback when sub-stepping produces NaN/Inf."""
    bsz = xt.shape[0]
    dt_t = dt_val * torch.ones((bsz,), device=xt.device, dtype=xt.dtype).unsqueeze(-1).unsqueeze(-1)
    return xt - vt * dt_t


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _taylor_approx(
    order: int, diff: Tensor, v: Tensor,
    dv: Optional[Tensor], d2v: Optional[Tensor],
) -> Tensor:
    """Taylor expansion of velocity: v(t + diff) ≈ v + diff*dv + 0.5*diff²*d2v."""
    if order >= 1 and dv is not None:
        result = v + diff * dv
        if order >= 2 and d2v is not None:
            result = result + 0.5 * diff ** 2 * d2v
        return result
    return v


def _compute_derivatives(
    vt: Tensor, state: State, device: torch.device, dtype: torch.dtype,
) -> Tuple[Optional[Tensor], Optional[Tensor], int]:
    """Compute velocity derivatives from history using finite differences.

    Includes safety clamping: model outputs carry bfloat16 quantization noise
    (~0.8% relative error).  At small step sizes the finite-difference
    amplification can exceed the signal, so we fall back to lower order when
    the derivative contribution would be unreliable.

    Returns (dv, d2v, derivative_order_used).
    """
    history = state.get("velocity_history", [])
    if len(history) == 0:
        return None, None, 0

    # 1st order: forward difference
    v_prev, h_prev = history[-1]
    # Ensure v_prev is in the requested dtype (should be float32 already)
    v_prev = v_prev.to(dtype=dtype)
    h1 = torch.tensor(h_prev, device=device, dtype=dtype)
    dv = (v_prev - vt) / h1

    # Safety: clamp 1st derivative if noise-dominated
    vt_rms = vt.norm() / max(vt.numel() ** 0.5, 1.0)
    dv_rms = dv.norm() / max(dv.numel() ** 0.5, 1.0)
    if vt_rms > 0 and dv_rms * abs(h_prev) > 5.0 * vt_rms:
        return None, None, 0

    if len(history) < 2:
        return dv, None, 1

    # 2nd order: three-point formula
    v_prev2, h_prev2 = history[-2]
    v_prev2 = v_prev2.to(dtype=dtype)
    h2 = torch.tensor(h_prev2, device=device, dtype=dtype)
    d2v = 2.0 / (h1 * h2 * (h1 + h2)) * (
        v_prev2 * h1 - v_prev * (h1 + h2) + vt * h2
    )

    # Safety: clamp 2nd derivative similarly
    d2v_rms = d2v.norm() / max(d2v.numel() ** 0.5, 1.0)
    if vt_rms > 0 and d2v_rms * h_prev ** 2 > 5.0 * vt_rms:
        return dv, None, 1

    return dv, d2v, 2


# ---------------------------------------------------------------------------
# STORK2: 2nd-order Runge-Kutta-Gegenbauer
# ---------------------------------------------------------------------------

def _b_coeff(j: int) -> float:
    """RKG2 recurrence coefficient b_j.

    Reference: Eq. from https://doi.org/10.1016/j.jcp.2020.109879
    """
    if j < 0:
        raise ValueError("b_coeff: j must be >= 0")
    if j == 0:
        return 1.0
    if j == 1:
        return 1.0 / 3.0
    return 4.0 * (j - 1) * (j + 4) / (3.0 * j * (j + 1) * (j + 2) * (j + 3))


def _rkg2_substep(
    xt_f: Tensor, vt_f: Tensor, s: int,
    t: float, t_next: float,
    deriv_order: int, dv: Optional[Tensor], d2v: Optional[Tensor],
) -> Tensor:
    """Run the RKG2 Chebyshev sub-stepping loop. Returns result in float32."""
    Y_j_2 = xt_f.clone()
    Y_j_1 = xt_f.clone()
    Y_j = xt_f.clone()
    dt_tensor = (t - t_next) * torch.ones_like(xt_f)

    for j in range(1, s + 1):
        if j > 1:
            if j == 2:
                fraction = 4.0 / (3.0 * (s ** 2 + s - 2))
            else:
                fraction = ((j - 1) ** 2 + (j - 1) - 2) / (s ** 2 + s - 2)

        if j == 1:
            mu_tilde = 6.0 / ((s + 4) * (s - 1))
            Y_j = Y_j_1 - dt_tensor * mu_tilde * vt_f
        else:
            mu = (2 * j + 1) * _b_coeff(j) / (j * _b_coeff(j - 1))
            nu = -(j + 1) * _b_coeff(j) / (j * _b_coeff(j - 2))
            mu_tilde_j = mu * 6.0 / ((s + 4) * (s - 1))
            gamma_tilde = -mu_tilde_j * (1.0 - j * (j + 1) * _b_coeff(j - 1) / 2.0)

            diff = -fraction * (t - t_next) * torch.ones_like(xt_f)
            velocity = _taylor_approx(deriv_order, diff, vt_f, dv, d2v)
            Y_j = (mu * Y_j_1 + nu * Y_j_2 + (1.0 - mu - nu) * xt_f
                    - dt_tensor * mu_tilde_j * velocity
                    - dt_tensor * gamma_tilde * vt_f)

        Y_j_2 = Y_j_1
        Y_j_1 = Y_j

    return Y_j


def stork2_step(
    xt: Tensor, vt: Tensor, t_curr: float, t_prev: float,
    state: State, model_fn: Optional[ModelFn] = None,
) -> Tuple[Tensor, State]:
    """STORK2: 2nd-order Stabilized Taylor Orthogonal Runge-Kutta. 1 NFE/step.

    Uses Runge-Kutta-Gegenbauer sub-stepping with Taylor-approximated velocity.
    Step 0 bootstraps with Euler.  Subsequent steps use velocity history for
    derivative estimation and adaptive sub-steps for stiffness handling.

    If sub-stepping produces NaN/Inf, sub-steps are halved until stable.
    Falls back to Euler as a last resort.

    Configure via state:
        state["stork_substeps"] = 10  (default)
    """
    device = xt.device
    dtype = xt.dtype
    step_idx = state.get("step_index", 0)
    s_requested = state.get("stork_substeps", DEFAULT_SUBSTEPS)
    dt_val = t_curr - t_prev  # positive (t goes 1 -> 0)

    # --- Bootstrap: step 0 uses plain Euler ---
    if step_idx == 0:
        bsz = xt.shape[0]
        dt_t = dt_val * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        xt_next = xt - vt * dt_t

        state["step_index"] = 1
        state.setdefault("velocity_history", []).append((vt.float().clone(), dt_val))
        return xt_next, state

    # --- Main STORK2 step ---
    # Upcast to float32 for sub-stepping precision
    orig_dtype = xt.dtype
    xt_f = xt.float()
    vt_f = vt.float()

    dv, d2v, deriv_order = _compute_derivatives(vt_f, state, device, torch.float32)

    # Adaptive sub-stepping: try requested s, halve on NaN until s=2
    s = max(s_requested, 2)
    xt_next = None
    while s >= 2:
        result = _rkg2_substep(xt_f, vt_f, s, t_curr, t_prev, deriv_order, dv, d2v)
        candidate = result.to(orig_dtype)
        if torch.isfinite(candidate).all():
            xt_next = candidate
            break
        s = s // 2

    if xt_next is None:
        # All sub-step counts produced NaN — fall back to Euler
        xt_next = _euler_fallback(xt, vt, dt_val)
        logger.warning(
            f"[STORK2] step {step_idx} t={t_curr:.4f}→{t_prev:.4f}: "
            f"all sub-steps NaN (tried {s_requested}→2) → Euler"
        )

    # Update state
    state["step_index"] = step_idx + 1
    history = state.setdefault("velocity_history", [])
    history.append((vt_f.clone(), dt_val))
    if len(history) > 3:
        state["velocity_history"] = history[-3:]

    return xt_next, state


# ---------------------------------------------------------------------------
# STORK4: 4th-order ROCK4 Runge-Kutta-Chebyshev
# ---------------------------------------------------------------------------

def _coeff_rock1(j: int) -> float:
    """ROCK1 (1st-order) recurrence coefficient."""
    return 2.0 / ((j + 1) * (j + 2))


def _mdegr(s: int, ms):
    """Find optimal degree in precomputed table for ROCK4.

    Returns (mdeg, (mz_index, mr_pointer)).
    """
    mp0 = 0  # index into ms
    mp1 = 1  # recf pointer
    mdeg = s
    for i in range(len(ms)):
        if (ms[i] / s) >= 1.0:
            mdeg = int(ms[i])
            mp0 = i
            mp1 = mp1 - 1
            break
        else:
            mp1 = mp1 + int(ms[i]) * 2 - 1
    return mdeg, int(mp0), int(mp1)


def _rock4_substep(
    xt_f: Tensor, vt_f: Tensor, s: int,
    t: float, t_next: float,
    deriv_order: int, dv: Optional[Tensor], d2v: Optional[Tensor],
    MS, FPA, FPB, RECF,
) -> Tensor:
    """Run the ROCK4 Chebyshev sub-stepping loop. Returns result in float32."""
    # Clamp to max supported ROCK4 degree
    MAX_ROCK4 = int(MS.max())
    if s > MAX_ROCK4:
        s = MAX_ROCK4

    mdeg, mz, mr = _mdegr(s, MS)

    # Part 1: ROCK4 sub-stepping
    Y_j_2 = xt_f.clone()
    Y_j_1 = xt_f.clone()
    Y_j = xt_f.clone()

    t_start = t * torch.ones_like(xt_f)
    ci1 = t_start.clone()
    ci2 = t_start.clone()
    ci3 = t_start.clone()

    for j in range(1, mdeg + 1):
        if j == 1:
            temp1 = -(t - t_next) * RECF[mr] * torch.ones_like(xt_f)
            ci1 = t_start + temp1
            ci2 = ci1.clone()
            Y_j_1 = xt_f + temp1 * vt_f
        else:
            diff = ci1 - t_start
            velocity = _taylor_approx(deriv_order, diff, vt_f, dv, d2v)

            temp1 = -(t - t_next) * RECF[mr + 2 * (j - 2) + 1] * torch.ones_like(xt_f)
            temp3 = -RECF[mr + 2 * (j - 2) + 2] * torch.ones_like(xt_f)
            temp2 = torch.ones_like(xt_f) - temp3

            ci1 = temp1 + temp2 * ci2 + temp3 * ci3
            Y_j = temp1 * velocity + temp2 * Y_j_1 + temp3 * Y_j_2

        Y_j_2 = Y_j_1.clone()
        Y_j_1 = Y_j.clone() if j > 1 else Y_j_1
        ci3 = ci2.clone()
        ci2 = ci1.clone()

    # Part 2: ROCK4 finishing four-step procedure
    # Step 1
    temp1 = -(t - t_next) * FPA[mz, 0] * torch.ones_like(xt_f)
    diff = ci1 - t_start
    velocity = _taylor_approx(deriv_order, diff, vt_f, dv, d2v)
    F1 = velocity
    Y_j_3 = Y_j + temp1 * F1

    # Step 2
    ci2_f = ci1 + temp1
    temp1_2 = -(t - t_next) * FPA[mz, 1] * torch.ones_like(xt_f)
    temp2_2 = -(t - t_next) * FPA[mz, 2] * torch.ones_like(xt_f)
    diff = ci2_f - t_start
    velocity = _taylor_approx(deriv_order, diff, vt_f, dv, d2v)
    F2 = velocity
    Y_j_4 = Y_j + temp1_2 * F1 + temp2_2 * F2

    # Step 3
    ci2_f = ci1 + temp1_2 + temp2_2
    temp1_3 = -(t - t_next) * FPA[mz, 3] * torch.ones_like(xt_f)
    temp2_3 = -(t - t_next) * FPA[mz, 4] * torch.ones_like(xt_f)
    temp3_3 = -(t - t_next) * FPA[mz, 5] * torch.ones_like(xt_f)
    diff = ci2_f - t_start
    velocity = _taylor_approx(deriv_order, diff, vt_f, dv, d2v)
    F3 = velocity

    # Step 4 (final)
    ci2_f = ci1 + temp1_3 + temp2_3 + temp3_3
    temp1_4 = -(t - t_next) * FPB[mz, 0] * torch.ones_like(xt_f)
    temp2_4 = -(t - t_next) * FPB[mz, 1] * torch.ones_like(xt_f)
    temp3_4 = -(t - t_next) * FPB[mz, 2] * torch.ones_like(xt_f)
    temp4_4 = -(t - t_next) * FPB[mz, 3] * torch.ones_like(xt_f)
    diff = ci2_f - t_start
    velocity = _taylor_approx(deriv_order, diff, vt_f, dv, d2v)
    F4 = velocity

    return Y_j + temp1_4 * F1 + temp2_4 * F2 + temp3_4 * F3 + temp4_4 * F4


def stork4_step(
    xt: Tensor, vt: Tensor, t_curr: float, t_prev: float,
    state: State, model_fn: Optional[ModelFn] = None,
) -> Tuple[Tensor, State]:
    """STORK4: 4th-order Stabilized Taylor Orthogonal Runge-Kutta. 1 NFE/step.

    Uses ROCK4 sub-stepping with Taylor-approximated velocity and precomputed
    Chebyshev coefficients.  Step 0 bootstraps with Euler.

    Adaptive sub-stepping: if sub-stepping produces NaN/Inf, sub-steps are
    halved until stable.  Falls back to Euler as a last resort.

    Configure via state:
        state["stork_substeps"] = 10  (default)
    """
    from acestep.core.generation.stork4_constants import MS, FPA, FPB, RECF

    device = xt.device
    dtype = xt.dtype
    step_idx = state.get("step_index", 0)
    s_requested = state.get("stork_substeps", DEFAULT_SUBSTEPS)

    # ROCK4 coefficient max
    MAX_ROCK4 = int(MS.max())
    s_requested = min(s_requested, MAX_ROCK4)

    dt_val = t_curr - t_prev

    # --- Bootstrap: step 0 uses plain Euler ---
    if step_idx == 0:
        bsz = xt.shape[0]
        dt_t = dt_val * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        xt_next = xt - vt * dt_t

        state["step_index"] = 1
        state.setdefault("velocity_history", []).append((vt.float().clone(), dt_val))
        return xt_next, state

    # --- Main STORK4 step ---
    orig_dtype = xt.dtype
    xt_f = xt.float()
    vt_f = vt.float()

    dv, d2v, deriv_order = _compute_derivatives(vt_f, state, device, torch.float32)

    # Adaptive sub-stepping: try requested s, halve on NaN until s=2
    s = max(s_requested, 2)
    xt_next = None
    while s >= 2:
        result = _rock4_substep(
            xt_f, vt_f, s, t_curr, t_prev,
            deriv_order, dv, d2v,
            MS, FPA, FPB, RECF,
        )
        candidate = result.to(orig_dtype)
        if torch.isfinite(candidate).all():
            xt_next = candidate
            break
        s = s // 2

    if xt_next is None:
        xt_next = _euler_fallback(xt, vt, dt_val)
        logger.warning(
            f"[STORK4] step {step_idx} t={t_curr:.4f}→{t_prev:.4f}: "
            f"all sub-steps NaN (tried {s_requested}→2) → Euler"
        )

    # Update state
    state["step_index"] = step_idx + 1
    history = state.setdefault("velocity_history", [])
    history.append((vt_f.clone(), dt_val))
    if len(history) > 3:
        state["velocity_history"] = history[-3:]

    return xt_next, state
