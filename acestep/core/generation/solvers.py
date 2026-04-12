"""
Solver registry for ACE-Step flow matching diffusion.

Each solver integrates the ODE  dx/dt = -v(x,t)  from t=1 (noise) to t=0 (clean).
The model predicts velocity v(x,t) at each step, and the solver advances x_t.

Solver interface:
    solver_step(xt, vt, t_curr, t_prev, state, model_fn=None) -> (xt_next, state)

    - xt:        Current latent tensor [bsz, seq, dim]
    - vt:        Velocity prediction at (xt, t_curr), already CFG-processed [bsz, seq, dim]
    - t_curr:    Current timestep (float, 1.0 = noise)
    - t_prev:    Next timestep (float, 0.0 = clean)
    - state:     Mutable dict for multi-step solvers (velocity history, etc.)
    - model_fn:  Optional callback for multi-evaluation solvers:
                 model_fn(xt, t) -> vt  (returns CFG-processed velocity)
    Returns:     (xt_next, state)

To add a new solver:
    1. Define a step function following the interface above
    2. Register it in the SOLVERS dict with a short name
    3. Add the name to SOLVER_INFO for metadata (display name, NFE, order)
"""

import torch
from typing import Any, Callable, Dict, Optional, Tuple

from acestep.core.generation.jkass_solvers import jkass_quality_step, jkass_fast_step
from acestep.core.generation.stork_solver import stork2_step, stork4_step
from acestep.core.generation.dopri_solvers import rk5_step, dopri5_step, dop853_step

Tensor = torch.Tensor
State = Dict[str, Any]
ModelFn = Callable[[Tensor, float], Tensor]


# ---------------------------------------------------------------------------
# Solver step functions
# ---------------------------------------------------------------------------

def euler_step(
    xt: Tensor, vt: Tensor, t_curr: float, t_prev: float,
    state: State, model_fn: Optional[ModelFn] = None,
) -> Tuple[Tensor, State]:
    """Euler method (1st order). 1 NFE per step."""
    dt = t_curr - t_prev
    bsz = xt.shape[0]
    dt_tensor = dt * torch.ones((bsz,), device=xt.device, dtype=xt.dtype).unsqueeze(-1).unsqueeze(-1)
    xt_next = xt - vt * dt_tensor
    return xt_next, state


def heun_step(
    xt: Tensor, vt: Tensor, t_curr: float, t_prev: float,
    state: State, model_fn: Optional[ModelFn] = None,
) -> Tuple[Tensor, State]:
    """Heun's method (2nd order). 2 NFE per step.
    
    1. Euler predictor:  x_pred = x_t - v(x_t, t) * dt
    2. Evaluate:         v_pred = model(x_pred, t_prev)  
    3. Corrector:        x_next = x_t - 0.5 * (v_t + v_pred) * dt
    """
    if model_fn is None:
        # Fallback to Euler if no model callback
        return euler_step(xt, vt, t_curr, t_prev, state, model_fn)

    dt = t_curr - t_prev
    bsz = xt.shape[0]
    dt_tensor = dt * torch.ones((bsz,), device=xt.device, dtype=xt.dtype).unsqueeze(-1).unsqueeze(-1)

    # Predictor: Euler step
    x_pred = xt - vt * dt_tensor

    # Evaluate velocity at predicted point
    v_pred = model_fn(x_pred, t_prev)

    # Corrector: average of both velocities
    xt_next = xt - 0.5 * (vt + v_pred) * dt_tensor
    return xt_next, state


def dpm_pp_2m_step(
    xt: Tensor, vt: Tensor, t_curr: float, t_prev: float,
    state: State, model_fn: Optional[ModelFn] = None,
) -> Tuple[Tensor, State]:
    """DPM++ 2M (2nd order multistep). 1 NFE per step.
    
    Uses the previous step's velocity for second-order correction,
    so no extra model calls needed. First step falls back to Euler.
    """
    dt = t_curr - t_prev
    bsz = xt.shape[0]
    dt_tensor = dt * torch.ones((bsz,), device=xt.device, dtype=xt.dtype).unsqueeze(-1).unsqueeze(-1)

    prev_vt = state.get("prev_vt")
    if prev_vt is not None:
        # Second-order: blend current and previous velocity (Adams-Bashforth style)
        # For uniform steps: x_next = x_t - dt * (3/2 * v_t - 1/2 * v_{t-1})
        vt_corrected = 1.5 * vt - 0.5 * prev_vt
    else:
        # First step: plain Euler
        vt_corrected = vt

    state["prev_vt"] = vt.clone()
    xt_next = xt - vt_corrected * dt_tensor
    return xt_next, state


def dpm_pp_3m_step(
    xt: Tensor, vt: Tensor, t_curr: float, t_prev: float,
    state: State, model_fn: Optional[ModelFn] = None,
) -> Tuple[Tensor, State]:
    """DPM++ 3M (3rd order multistep). 1 NFE per step.

    Uses the two most-recent velocity predictions for a 3rd-order
    Adams-Bashforth correction.  Falls back to AB2 on step 2 and
    Euler on step 1.

    Coefficients (uniform-step AB3):
        v_eff = (23/12)*v_n - (16/12)*v_{n-1} + (5/12)*v_{n-2}

    This mirrors how DPM++ 2M uses fixed AB2 coefficients (3/2, -1/2)
    which work well even with non-uniform timestep schedules.
    """
    dt = t_curr - t_prev
    bsz = xt.shape[0]
    dt_tensor = dt * torch.ones((bsz,), device=xt.device, dtype=xt.dtype).unsqueeze(-1).unsqueeze(-1)

    prev_vt = state.get("prev_vt")
    prev_prev_vt = state.get("prev_prev_vt")

    if prev_vt is not None and prev_prev_vt is not None:
        # Third-order: Adams-Bashforth 3
        vt_corrected = (23.0 / 12.0) * vt - (16.0 / 12.0) * prev_vt + (5.0 / 12.0) * prev_prev_vt
    elif prev_vt is not None:
        # Second-order: AB2 (same as DPM++ 2M)
        vt_corrected = 1.5 * vt - 0.5 * prev_vt
    else:
        # First step: plain Euler
        vt_corrected = vt

    # Update state: shift history
    state["prev_prev_vt"] = state.get("prev_vt")
    state["prev_vt"] = vt.clone()

    xt_next = xt - vt_corrected * dt_tensor
    return xt_next, state


def rk4_step(
    xt: Tensor, vt: Tensor, t_curr: float, t_prev: float,
    state: State, model_fn: Optional[ModelFn] = None,
) -> Tuple[Tensor, State]:
    """Runge-Kutta 4th order. 4 NFE per step (3 extra evaluations).
    
    Classic RK4 with intermediate evaluations at midpoint and endpoint.
    Extremely accurate but 4x the cost of Euler.
    """
    if model_fn is None:
        # Fallback to Euler if no model callback
        return euler_step(xt, vt, t_curr, t_prev, state, model_fn)

    dt = t_curr - t_prev
    bsz = xt.shape[0]
    dt_tensor = dt * torch.ones((bsz,), device=xt.device, dtype=xt.dtype).unsqueeze(-1).unsqueeze(-1)
    t_mid = (t_curr + t_prev) / 2.0

    # k1 = v(x_t, t_curr) — already computed
    k1 = vt

    # k2 = v(x_t - 0.5*dt*k1, t_mid)
    x2 = xt - 0.5 * dt_tensor * k1
    k2 = model_fn(x2, t_mid)

    # k3 = v(x_t - 0.5*dt*k2, t_mid)
    x3 = xt - 0.5 * dt_tensor * k2
    k3 = model_fn(x3, t_mid)

    # k4 = v(x_t - dt*k3, t_prev)
    x4 = xt - dt_tensor * k3
    k4 = model_fn(x4, t_prev)

    # Weighted average
    xt_next = xt - (dt_tensor / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return xt_next, state


def dpm_pp_2m_ada_step(
    xt: Tensor, vt: Tensor, t_curr: float, t_prev: float,
    state: State, model_fn: Optional[ModelFn] = None,
) -> Tuple[Tensor, State]:
    """DPM++ 2M with step-ratio-corrected Adams-Bashforth coefficients.

    The standard DPM++ 2M uses fixed AB2 coefficients (3/2, −1/2) which
    are optimal for uniform timestep spacing.  With non-uniform schedules
    (Karras, beta57, cosine, etc.) these become suboptimal.

    This variant dynamically computes the AB2 weights from the ratio of
    current to previous step size::

        r  = dt_curr / dt_prev
        c₁ = 1 + r/2
        c₀ = r/2
        v_eff = c₁ · v_n − c₀ · v_{n-1}

    For uniform steps (r = 1) this reduces to the standard formula.
    Still only 1 NFE per step.
    """
    dt = t_curr - t_prev
    bsz = xt.shape[0]
    dt_tensor = dt * torch.ones((bsz,), device=xt.device, dtype=xt.dtype).unsqueeze(-1).unsqueeze(-1)

    prev_vt = state.get("prev_vt")
    prev_dt = state.get("prev_dt")

    if prev_vt is not None and prev_dt is not None and prev_dt > 0:
        # Step-ratio-corrected AB2
        r = dt / prev_dt
        c1 = 1.0 + r / 2.0   # weight for current velocity
        c0 = r / 2.0          # weight for previous velocity (subtracted)
        vt_corrected = c1 * vt - c0 * prev_vt
    else:
        # First step: plain Euler
        vt_corrected = vt

    state["prev_vt"] = vt.clone()
    state["prev_dt"] = dt
    xt_next = xt - vt_corrected * dt_tensor
    return xt_next, state


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SOLVERS = {
    "euler": euler_step,
    "ode": euler_step,       # alias for backward compatibility
    "heun": heun_step,
    "dpm2m": dpm_pp_2m_step,
    "dpm2m_ada": dpm_pp_2m_ada_step,
    "dpm3m": dpm_pp_3m_step,
    "rk4": rk4_step,
    "rk5": rk5_step,
    "dopri5": dopri5_step,
    "dop853": dop853_step,
    "jkass_quality": jkass_quality_step,
    "jkass_fast": jkass_fast_step,
    "stork2": stork2_step,
    "stork4": stork4_step,
}

# Metadata for each solver
SOLVER_INFO = {
    "euler":   {"name": "Euler",       "order": 1, "nfe": 1,  "needs_model_fn": False},
    "ode":     {"name": "Euler (ODE)", "order": 1, "nfe": 1,  "needs_model_fn": False},
    "heun":    {"name": "Heun",        "order": 2, "nfe": 2,  "needs_model_fn": True},
    "dpm2m":   {"name": "DPM++ 2M",    "order": 2, "nfe": 1,  "needs_model_fn": False},
    "dpm2m_ada": {"name": "DPM++ 2M Adaptive", "order": 2, "nfe": 1, "needs_model_fn": False},
    "dpm3m":   {"name": "DPM++ 3M",    "order": 3, "nfe": 1,  "needs_model_fn": False},
    "rk4":     {"name": "RK4",         "order": 4, "nfe": 4,  "needs_model_fn": True},
    "rk5":     {"name": "RK5 (DOPRI5)", "order": 5, "nfe": 6, "needs_model_fn": True},
    "dopri5":  {"name": "DOPRI5 Adaptive", "order": 5, "nfe": 7, "needs_model_fn": True},
    "dop853":  {"name": "DOP853",      "order": 8, "nfe": 13, "needs_model_fn": True},
    "jkass_quality": {"name": "JKASS Quality", "order": 2, "nfe": 2, "needs_model_fn": True},
    "jkass_fast":    {"name": "JKASS Fast",    "order": 1, "nfe": 1, "needs_model_fn": False},
    "stork2":  {"name": "STORK 2",    "order": 2, "nfe": 1,  "needs_model_fn": False},
    "stork4":  {"name": "STORK 4",    "order": 4, "nfe": 1,  "needs_model_fn": False},
}

# All valid solver names (for validation)
VALID_SOLVERS = set(SOLVERS.keys())


def get_solver(name: str):
    """Get a solver step function by name.
    
    Returns:
        (solver_fn, needs_model_fn: bool)
    
    Raises:
        ValueError if the solver name is not recognized.
    """
    if name not in SOLVERS:
        valid = ", ".join(sorted(VALID_SOLVERS - {"ode"}))  # hide ode alias
        raise ValueError(f"Unknown solver '{name}'. Valid solvers: {valid}")
    info = SOLVER_INFO.get(name, {})
    return SOLVERS[name], info.get("needs_model_fn", False)
