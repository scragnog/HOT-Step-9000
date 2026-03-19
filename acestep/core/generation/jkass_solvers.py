"""
JKASS Solvers for ACE-Step flow matching diffusion.

Ported from jeankassio/JK-AceStep-Nodes, adapted to our
solver interface: solver_step(xt, vt, t_curr, t_prev, state, model_fn) -> (xt_next, state)

Two variants:
  - jkass_quality: Heun with derivative averaging (2 NFE). Highest accuracy.
  - jkass_fast: Euler with momentum + frequency damping + temporal smoothing (1 NFE).
"""

import torch
from typing import Any, Callable, Dict, Optional, Tuple

Tensor = torch.Tensor
State = Dict[str, Any]
ModelFn = Callable[[Tensor, float], Tensor]


# ---------------------------------------------------------------------------
# Helper functions (applied to velocity delta during integration)
# ---------------------------------------------------------------------------

def _apply_frequency_damping(tensor: Tensor, damping: float) -> Tensor:
    """Exponential decay across higher frequency bins (last axis).

    tensor: expected shape [B, seq, dim] where dim ~ frequency.
    damping: > 0 => stronger attenuation of higher freq bins.
    """
    if damping <= 0:
        return tensor
    F = tensor.shape[-1]
    freqs = torch.linspace(0.0, 1.0, F, device=tensor.device, dtype=tensor.dtype)
    freq_mult = torch.exp(-damping * (freqs ** 2))
    return tensor * freq_mult


def _apply_temporal_smoothing(tensor: Tensor, strength: float) -> Tensor:
    """Tiny temporal smoothing kernel [0.25, 0.50, 0.25] across axis=1 (sequence/time).

    strength: 0.0 = none, 1.0 = full smoothing.
    """
    if strength <= 0:
        return tensor
    if tensor.dim() < 3:
        return tensor
    # tensor: [B, T, D]
    B, T, D = tensor.shape
    if T < 3:
        return tensor

    # Build depthwise 1D kernel across time dim
    # Reshape to [B, D, T] for conv1d, then back
    x = tensor.permute(0, 2, 1)  # [B, D, T]
    kernel = torch.tensor([0.25, 0.5, 0.25], dtype=tensor.dtype, device=tensor.device)
    kernel = kernel.view(1, 1, 3).expand(D, 1, 3)
    padded = torch.nn.functional.pad(x, (1, 1), mode='reflect')
    smoothed = torch.nn.functional.conv1d(padded, kernel, groups=D)
    smoothed = smoothed.permute(0, 2, 1)  # back to [B, T, D]
    return (1.0 - strength) * tensor + strength * smoothed


# ---------------------------------------------------------------------------
# Solver step functions
# ---------------------------------------------------------------------------

def jkass_quality_step(
    xt: Tensor, vt: Tensor, t_curr: float, t_prev: float,
    state: State, model_fn: Optional[ModelFn] = None,
) -> Tuple[Tensor, State]:
    """JKASS Quality: Heun with derivative averaging. 2 NFE per step.

    Uses the model_fn callback for a second evaluation at the predicted point,
    then averages the two derivatives for higher accuracy.
    Falls back to Euler if model_fn is unavailable or on the last step.
    """
    dt = t_curr - t_prev
    bsz = xt.shape[0]
    dt_tensor = dt * torch.ones((bsz,), device=xt.device, dtype=xt.dtype).unsqueeze(-1).unsqueeze(-1)

    if model_fn is None or t_prev <= 0:
        # Fallback to Euler
        xt_next = xt - vt * dt_tensor
        return xt_next, state

    # Euler predictor
    x_pred = xt - vt * dt_tensor

    # Second evaluation at predicted point
    v_pred = model_fn(x_pred, t_prev)

    # Average the two derivatives (Heun's method)
    v_avg = (vt + v_pred) / 2.0
    xt_next = xt - v_avg * dt_tensor

    return xt_next, state


def jkass_fast_step(
    xt: Tensor, vt: Tensor, t_curr: float, t_prev: float,
    state: State, model_fn: Optional[ModelFn] = None,
) -> Tuple[Tensor, State]:
    """JKASS Fast: Euler with momentum + frequency damping + temporal smoothing. 1 NFE.

    Extra kwargs are read from state:
      - state["beat_stability"]: float, blend previous delta with current (0=off, 1=full prev)
      - state["frequency_damping"]: float, exponential decay on high-freq bins
      - state["temporal_smoothing"]: float, tiny temporal kernel on delta
    """
    dt = t_curr - t_prev
    bsz = xt.shape[0]
    dt_tensor = dt * torch.ones((bsz,), device=xt.device, dtype=xt.dtype).unsqueeze(-1).unsqueeze(-1)

    # Read tuning parameters from state
    beat_stability = state.get("beat_stability", 0.0)
    frequency_damping = state.get("frequency_damping", 0.0)
    temporal_smoothing = state.get("temporal_smoothing", 0.0)

    # Compute velocity delta
    delta = vt

    # Beat stability: momentum blending with previous step's delta
    prev_delta = state.get("prev_delta")
    if prev_delta is not None and beat_stability > 0:
        delta = (1.0 - beat_stability) * delta + beat_stability * prev_delta
    state["prev_delta"] = delta.clone()

    # Frequency damping: exponential decay on high-freq bins
    if frequency_damping > 0:
        delta = _apply_frequency_damping(delta, frequency_damping)

    # Temporal smoothing: tiny kernel across time axis
    if temporal_smoothing > 0:
        delta = _apply_temporal_smoothing(delta, temporal_smoothing)

    xt_next = xt - delta * dt_tensor
    return xt_next, state
