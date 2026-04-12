"""
Higher-order explicit Runge-Kutta ODE solvers for ACE-Step flow matching.

Implements:
  - rk5_step:     Dormand-Prince 5th order, fixed-step  (6 NFE)
  - dopri5_step:  Dormand-Prince 5(4) adaptive with sub-stepping  (6-7+ NFE)
  - dop853_step:  Dormand-Prince 8th order, fixed-step  (13 NFE)

Coefficient sources:
  - DOPRI5: JAX (google/jax) experimental.ode, diffrax (patrick-kidger/diffrax)
  - DOP853: diffrax, originally from Prince & Dormand (1981)

Each solver follows the standard interface:
    solver_step(xt, vt, t_curr, t_prev, state, model_fn=None) -> (xt_next, state)
"""

import torch
from typing import Any, Callable, Dict, List, Optional, Tuple

Tensor = torch.Tensor
State = Dict[str, Any]
ModelFn = Callable[[Tensor, float], Tensor]


# ═══════════════════════════════════════════════════════════════════════════
# DOPRI5 Butcher tableau — Dormand-Prince 5th order (7 stages, FSAL)
# Verified from JAX (google/jax) and diffrax (patrick-kidger/diffrax)
# ═══════════════════════════════════════════════════════════════════════════

_DOPRI5_C = [1/5, 3/10, 4/5, 8/9, 1.0, 1.0]  # nodes for stages 2..7

_DOPRI5_A = [
    # Stage 2
    [1/5],
    # Stage 3
    [3/40, 9/40],
    # Stage 4
    [44/45, -56/15, 32/9],
    # Stage 5
    [19372/6561, -25360/2187, 64448/6561, -212/729],
    # Stage 6
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
    # Stage 7 (FSAL — needed for adaptive error estimation only)
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
]

_DOPRI5_B = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]

# Error coefficients: b_5th − b_4th  (verified from JAX)
_DOPRI5_E = [
    35/384 - 1951/21600,       # k1
    0,                          # k2
    500/1113 - 22642/50085,    # k3
    125/192 - 451/720,          # k4
    -2187/6784 - (-12231/42400),  # k5
    11/84 - 649/6300,           # k6
    -1.0/60.0,                  # k7
]


# ═══════════════════════════════════════════════════════════════════════════
# DOP853 Butcher tableau — Dormand-Prince 8th order
# 13 stages needed for the 8th-order solution (k1 + 12 extra).
# Verified from diffrax (patrick-kidger/diffrax), originally from
# Prince & Dormand (1981): "High order embedded Runge-Kutta formulae"
# ═══════════════════════════════════════════════════════════════════════════

_DOP853_C = [
    1/18, 1/12, 1/8, 5/16, 3/8,
    59/400, 93/200, 5490023248/9719169821,
    13/20, 1201146811/1299019798,
    1.0, 1.0,
]

_DOP853_A = [
    # Stage 2
    [1/18],
    # Stage 3
    [1/48, 1/16],
    # Stage 4
    [1/32, 0, 3/32],
    # Stage 5
    [5/16, 0, -75/64, 75/64],
    # Stage 6
    [3/80, 0, 0, 3/16, 3/20],
    # Stage 7
    [29443841/614563906, 0, 0, 77736538/692538347,
     -28693883/1125000000, 23124283/1800000000],
    # Stage 8
    [16016141/946692911, 0, 0, 61564180/158732637,
     22789713/633445777, 545815736/2771057229, -180193667/1043307555],
    # Stage 9
    [39632708/573591083, 0, 0, -433636366/683701615,
     -421739975/2616292301, 100302831/723423059,
     790204164/839813087, 800635310/3783071287],
    # Stage 10
    [246121993/1340847787, 0, 0, -37695042795/15268766246,
     -309121744/1061227803, -12992083/490766935,
     6005943493/2108947869, 393006217/1396673457, 123872331/1001029789],
    # Stage 11
    [-1028468189/846180014, 0, 0, 8478235783/508512852,
     1311729495/1432422823, -10304129995/1701304382,
     -48777925059/3047939560, 15336726248/1032824649,
     -45442868181/3398467696, 3065993473/597172653],
    # Stage 12
    [185892177/718116043, 0, 0, -3185094517/667107341,
     -477755414/1098053517, -703635378/230739211,
     5731566787/1027545527, 5232866602/850066563,
     -4093664535/808688257, 3962137247/1805957418, 65686358/487910083],
    # Stage 13
    [403863854/491063109, 0, 0, -5068492393/434740067,
     -411421997/543043805, 652783627/914296604,
     11173962825/925320556, -13158990841/6184727034,
     3936647629/1978049680, -160528059/685178525,
     248638103/1413531060, 0],
]

_DOP853_B = [
    14005451/335480064,        # k1
    0,                          # k2
    0,                          # k3
    0,                          # k4
    0,                          # k5
    -59238493/1068277825,       # k6
    181606767/758867731,        # k7
    561292985/797845732,        # k8
    -1041891430/1371343529,     # k9
    760417239/1151165299,       # k10
    118820643/751138087,        # k11
    -528747749/2220607170,      # k12
    1/4,                        # k13
]


# ═══════════════════════════════════════════════════════════════════════════
# Generic explicit Runge-Kutta step
# ═══════════════════════════════════════════════════════════════════════════

def _erk_step(
    xt: Tensor, vt: Tensor, t_curr: float, dt: float,
    a_lower: List[List[float]], b_sol: List[float],
    c_nodes: List[float], model_fn: ModelFn,
) -> Tuple[Tensor, List[Tensor]]:
    """Generic explicit Runge-Kutta step using a Butcher tableau.

    Integrates  dx/dt = −v(x,t)  from ``t_curr`` to ``t_curr − dt``.

    Args:
        xt:       Current latent state [bsz, seq, dim].
        vt:       Velocity prediction at (xt, t_curr) — stage k1.
        t_curr:   Current timestep.
        dt:       Step size (positive; t_curr − t_prev).
        a_lower:  Lower-triangular coefficient rows for stages 2..N.
        b_sol:    Solution weights for all N stages.
        c_nodes:  Time-fraction nodes for stages 2..N.
        model_fn: Model callback — model_fn(x, t) → v.

    Returns:
        xt_next:  Updated latent state.
        ks:       All stage velocity evaluations [k1, k2, …, kN].
    """
    bsz = xt.shape[0]
    dt_t = dt * torch.ones(
        (bsz,), device=xt.device, dtype=xt.dtype,
    ).unsqueeze(-1).unsqueeze(-1)

    ks: List[Tensor] = [vt]  # k[0] = k1 (already evaluated)

    # ── Compute intermediate stages ──────────────────────────────
    for a_row, ci in zip(a_lower, c_nodes):
        combo = torch.zeros_like(xt)
        for j, aij in enumerate(a_row):
            if aij != 0:
                combo = combo + aij * ks[j]
        x_i = xt - dt_t * combo
        ks.append(model_fn(x_i, t_curr - ci * dt))

    # ── Combine stages with solution weights ─────────────────────
    sol = torch.zeros_like(xt)
    for j, bj in enumerate(b_sol):
        if bj != 0 and j < len(ks):
            sol = sol + bj * ks[j]
    xt_next = xt - dt_t * sol

    return xt_next, ks


# ═══════════════════════════════════════════════════════════════════════════
# RK5 — Dormand-Prince 5th order, fixed step
# ═══════════════════════════════════════════════════════════════════════════

def rk5_step(
    xt: Tensor, vt: Tensor, t_curr: float, t_prev: float,
    state: State, model_fn: Optional[ModelFn] = None,
) -> Tuple[Tensor, State]:
    """Dormand-Prince 5th order.  6 NFE per step (5 extra evaluations).

    Uses the DOPRI5 Butcher tableau for 5th-order accuracy.
    Significantly more accurate than RK4 at +50% cost.  Best used with
    fewer steps (20–30) for quality/speed trade-off.
    """
    if model_fn is None:
        from acestep.core.generation.solvers import euler_step
        return euler_step(xt, vt, t_curr, t_prev, state, model_fn)

    dt = t_curr - t_prev
    # Stages 2..6 only (5 extra evaluations) — skip the FSAL stage
    xt_next, _ = _erk_step(
        xt, vt, t_curr, dt,
        _DOPRI5_A[:5], _DOPRI5_B[:6], _DOPRI5_C[:5], model_fn,
    )
    return xt_next, state


# ═══════════════════════════════════════════════════════════════════════════
# DOPRI5 — Adaptive Dormand-Prince 5(4) with error-driven sub-stepping
# ═══════════════════════════════════════════════════════════════════════════

def dopri5_step(
    xt: Tensor, vt: Tensor, t_curr: float, t_prev: float,
    state: State, model_fn: Optional[ModelFn] = None,
) -> Tuple[Tensor, State]:
    """Adaptive Dormand-Prince 5(4) with sub-stepping and error control.

    Uses the embedded 4th-order solution to estimate local truncation error.
    If error exceeds tolerance the step interval is bisected automatically.
    When the schedule already provides fine-enough steps (the common case),
    this behaves identically to ``rk5_step`` with one extra NFE for error
    estimation.

    Tolerances can be set via state dict keys:
        ``dopri5_atol``  (default 1e-3)
        ``dopri5_rtol``  (default 1e-2)

    **7 NFE per accepted step** (6 for the solution + 1 FSAL for error).
    """
    if model_fn is None:
        from acestep.core.generation.solvers import euler_step
        return euler_step(xt, vt, t_curr, t_prev, state, model_fn)

    atol = state.get("dopri5_atol", 1e-3)
    rtol = state.get("dopri5_rtol", 1e-2)
    max_sub = state.get("dopri5_max_substeps", 8)
    safety = 0.9

    total_remaining = t_curr - t_prev
    t = t_curr
    x = xt
    v = vt
    h = total_remaining  # try the full interval first

    sub = 0
    while sub < max_sub and (t - t_prev) > 1e-10:
        h = min(h, t - t_prev)  # don't overshoot

        # Full DOPRI5 — all 6 a_lower rows including FSAL → k1..k7
        xt_next, ks = _erk_step(
            x, v, t, h,
            _DOPRI5_A, _DOPRI5_B, _DOPRI5_C, model_fn,
        )

        # ── Error estimate ───────────────────────────────────────
        bsz = x.shape[0]
        dt_t = h * torch.ones(
            (bsz,), device=x.device, dtype=x.dtype,
        ).unsqueeze(-1).unsqueeze(-1)

        error = torch.zeros_like(x)
        for j, ej in enumerate(_DOPRI5_E):
            if ej != 0 and j < len(ks):
                error = error + ej * ks[j]
        error = dt_t * error

        # RMS error norm (Hairer & Wanner formula)
        scale = atol + rtol * torch.max(torch.abs(x), torch.abs(xt_next))
        err_norm = torch.sqrt(torch.mean((error / scale) ** 2)).item()

        if err_norm <= 1.0:
            # ── Accept step ──────────────────────────────────────
            t = t - h
            x = xt_next
            v = ks[-1]  # FSAL: k7 becomes k1 of next sub-step
            # Grow step size for next sub-step (capped at 5×)
            if err_norm > 1e-10:
                h = h * min(5.0, safety * err_norm ** (-0.2))
            else:
                h = h * 5.0
        else:
            # ── Reject: shrink step (floored at 0.2×) ────────────
            h = h * max(0.2, safety * err_norm ** (-0.2))

        sub += 1

    return x, state


# ═══════════════════════════════════════════════════════════════════════════
# DOP853 — Dormand-Prince 8th order, fixed step
# ═══════════════════════════════════════════════════════════════════════════

def dop853_step(
    xt: Tensor, vt: Tensor, t_curr: float, t_prev: float,
    state: State, model_fn: Optional[ModelFn] = None,
) -> Tuple[Tensor, State]:
    """Dormand-Prince 8th order.  13 NFE per step (12 extra evaluations).

    Maximum-precision explicit solver.  Uses the DOP853 Butcher tableau
    from Prince & Dormand (1981) for 8th-order accuracy.  Extremely
    expensive — each step costs 13 model forward passes — but provides
    the highest ODE integration accuracy available.

    Best used with very few steps (5–15) when absolute precision matters
    more than generation speed.
    """
    if model_fn is None:
        from acestep.core.generation.solvers import euler_step
        return euler_step(xt, vt, t_curr, t_prev, state, model_fn)

    dt = t_curr - t_prev
    xt_next, _ = _erk_step(
        xt, vt, t_curr, dt,
        _DOP853_A, _DOP853_B, _DOP853_C, model_fn,
    )
    return xt_next, state
