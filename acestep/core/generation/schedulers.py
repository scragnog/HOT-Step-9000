"""
Scheduler registry for ACE-Step flow matching diffusion.

Schedulers define HOW timesteps are spaced between t=1 (noise) and t=0 (clean).
This is independent of the solver (which defines how each step is computed)
and the guidance mode (which defines how CFG is applied).

Scheduler interface:
    schedule_fn(num_steps: int, shift: float = 1.0) -> List[float]

    - num_steps:  Number of diffusion steps (N).
    - shift:      Timestep shift factor.  Applied as final warp:
                  t' = shift * t / (1 + (shift - 1) * t)

    Returns: List of N float values, descending, in (0, 1].
             Does NOT include a trailing 0.

To add a new scheduler:
    1. Define a function following the interface above
    2. Register it in the SCHEDULERS dict
    3. Add metadata to SCHEDULER_INFO
"""

import math
from typing import List, Optional


# ---------------------------------------------------------------------------
# Shift warp — shared by all schedulers
# ---------------------------------------------------------------------------

def _apply_shift(timesteps: List[float], shift: float) -> List[float]:
    """Apply the standard shift warp: t' = shift*t / (1 + (shift-1)*t).

    When shift == 1.0 this is an identity transform.
    """
    if shift == 1.0:
        return timesteps
    return [shift * t / (1.0 + (shift - 1.0) * t) for t in timesteps]


# ---------------------------------------------------------------------------
# Dynamic shift — auto-adjusts shift based on content properties
# ---------------------------------------------------------------------------

def compute_dynamic_shift(
    base_shift: float = 3.0,
    audio_duration: float = 60.0,
    num_steps: int = 30,
) -> float:
    """Compute an adaptive shift value based on generation context.

    Longer songs need more structural steps (higher shift), and fewer
    diffusion steps benefit from higher shift to concentrate on structure.
    Calibrated so that ``base_shift=3.0, duration=60s, steps=30`` returns
    exactly 3.0 — the current default for base models.

    The formula applies two multiplicative factors clamped to [0.8, 1.4]:

    **Duration factor** — ``1.0 + 0.15 * ((duration - 60) / 60)``
        Songs under 60 s get a slight shift decrease (more detail focus),
        songs over 60 s get a shift increase (more structural focus).

    **Step factor** — ``1.0 + 0.1 * ((30 - steps) / 30)``
        Fewer than 30 steps → higher shift (concentrate on structure).
        More than 30 steps → lower shift (budget allows detail).

    Args:
        base_shift: Starting shift value (typically 1.0–5.0).
        audio_duration: Target audio length in seconds.
        num_steps: Number of diffusion steps.

    Returns:
        Adjusted shift value, clamped to [1.0, 6.0].
    """
    # Duration factor: longer → more structural emphasis
    dur_factor = 1.0 + 0.15 * ((audio_duration - 60.0) / 60.0)
    dur_factor = max(0.8, min(1.4, dur_factor))

    # Step factor: fewer steps → concentrate on structure
    step_factor = 1.0 + 0.1 * ((30.0 - num_steps) / 30.0)
    step_factor = max(0.8, min(1.4, step_factor))

    adjusted = base_shift * dur_factor * step_factor
    return max(1.0, min(6.0, adjusted))


# ---------------------------------------------------------------------------
# Schedule functions
# ---------------------------------------------------------------------------

def linear_schedule(num_steps: int, shift: float = 1.0) -> List[float]:
    """Linear (uniform) spacing — the original ACE-Step default.

    Produces evenly-spaced timesteps from 1.0 toward 0.0.
    Equivalent to the existing ``torch.linspace(1, 0, N+1)[:-1]`` behaviour.
    """
    raw = [1.0 - i / num_steps for i in range(num_steps)]
    return _apply_shift(raw, shift)


def ddim_uniform_schedule(num_steps: int, shift: float = 1.0) -> List[float]:
    """Log-SNR uniform — uniform spacing in logit(t) space.

    In flow matching, SNR = (1-t)²/t², so log-SNR ∝ -logit(t).
    Uniform spacing in logit(t) concentrates steps where the
    signal-to-noise ratio changes most in *relative* terms.

    Gives an S-shaped distribution: dense around t=0.5, tapering
    at both extremes.  Preserves vocal detail budget.
    """
    # logit bounds: t=0.9986 → logit≈6.57,  t=0.0014 → logit≈-6.57
    t_max = 0.9986
    t_min = 0.0014
    logit_max = math.log(t_max / (1.0 - t_max))
    logit_min = math.log(t_min / (1.0 - t_min))

    raw = []
    for i in range(num_steps):
        frac = i / num_steps
        logit_t = logit_max + (logit_min - logit_max) * frac
        t = 1.0 / (1.0 + math.exp(-logit_t))  # sigmoid
        raw.append(t)

    raw = [max(min(t, 1.0), 1e-6) for t in raw]
    return _apply_shift(raw, shift)


def sgm_uniform_schedule(num_steps: int, shift: float = 1.0) -> List[float]:
    """Karras σ-ramp — uniform in σ^(1/ρ) space (EDM/Karras convention).

    Uses the well-tested Karras noise schedule ramp with ρ=7, adapted
    for flow matching.  Provides moderate front-loading: more steps in
    the structural region than linear, but not so extreme that detail
    is starved.
    """
    t_max = 0.999
    t_min = 0.001
    sigma_max = t_max / (1.0 - t_max)   # ≈999
    sigma_min = t_min / (1.0 - t_min)   # ≈0.001
    rho = 7.0  # Karras ramp parameter

    inv_rho = 1.0 / rho
    s_max = sigma_max ** inv_rho
    s_min = sigma_min ** inv_rho

    raw = []
    for i in range(num_steps):
        frac = i / num_steps
        sigma = (s_max + frac * (s_min - s_max)) ** rho
        t = sigma / (1.0 + sigma)
        raw.append(t)

    raw = [max(min(t, 1.0), 1e-6) for t in raw]
    return _apply_shift(raw, shift)


def bong_tangent_schedule(num_steps: int, shift: float = 1.0) -> List[float]:
    """Tangent-based spacing — concentrates steps at high noise levels.

    Uses a tangent curve to front-load steps where the model makes
    major structural decisions, with fewer steps in the fine-detail
    region near t=0.
    """
    raw = []
    for i in range(num_steps):
        # Map i to angle in (0, π/2), tangent maps (0,π/2) → (0, ∞)
        # We then normalise to (0, 1]
        frac = (i + 0.5) / num_steps  # avoid exact 0 and 1
        angle = frac * math.pi / 2.0
        tan_val = math.tan(angle)
        # Normalise: at frac=1, tan(π/2) → ∞, so use a bounded version
        # Use atan-based mapping: t = 1 - (2/π) * atan(tan_val * scale)
        # scale controls how aggressively front-loaded the schedule is
        scale = 1.5
        t = 1.0 - (2.0 / math.pi) * math.atan(tan_val * scale)
        raw.append(t)

    # Sort descending and clamp
    raw = sorted(raw, reverse=True)
    raw = [max(min(t, 1.0), 1e-6) for t in raw]
    return _apply_shift(raw, shift)


def linear_quadratic_schedule(num_steps: int, shift: float = 1.0) -> List[float]:
    """Linear start, quadratic finish — more steps for fine detail.

    The first half of steps are linearly spaced (even coverage of
    high-noise structural region), and the second half use quadratic
    spacing (denser toward t=0 for fine detail refinement).

    The crossover fraction (0.5) balances structure vs detail.
    """
    crossover = 0.5
    n_linear = max(int(num_steps * crossover), 1)
    n_quad = num_steps - n_linear

    # Linear region: from 1.0 down to crossover point
    t_cross = 1.0 - crossover  # e.g., 0.5
    linear_part = [1.0 - i * crossover / n_linear for i in range(n_linear)]

    # Quadratic region: from crossover point down toward 0
    quad_part = []
    for i in range(n_quad):
        frac = (i + 1) / n_quad  # 0 → 1, maps crossover → 0
        # Quadratic: denser at end (near 0)
        t = t_cross * (1.0 - frac ** 2)
        quad_part.append(t)

    raw = linear_part + quad_part
    raw = [max(min(t, 1.0), 1e-6) for t in raw]
    return _apply_shift(raw, shift)


def composite_schedule(
    num_steps: int,
    shift: float = 1.0,
    scheduler_a: str = "bong_tangent",
    scheduler_b: str = "linear",
    crossover: float = 0.5,
    split: float = 0.5,
) -> List[float]:
    """Two-stage composite — different schedulers for structure vs detail.

    Splits the total diffusion trajectory into two phases at a
    crossover timestep, using ``scheduler_a`` for the high-noise
    structural phase (t=1 → crossover) and ``scheduler_b`` for
    the low-noise detail phase (crossover → 0).

    Args:
        num_steps:   Total number of diffusion steps.
        shift:       Timestep shift factor (applied after compositing).
        scheduler_a: Scheduler name for the structural phase (t ≥ crossover).
        scheduler_b: Scheduler name for the detail phase (t < crossover).
        crossover:   Timestep value (0–1) where the phases meet. Default 0.5.
        split:       Fraction of total steps allocated to phase A. Default 0.5.
    """
    n_a = max(int(num_steps * split), 1)
    n_b = max(num_steps - n_a, 1)

    # Generate a full schedule from each scheduler, then filter to the
    # appropriate range.  We over-sample and pick the right density.
    fn_a = SCHEDULERS.get(scheduler_a, linear_schedule)
    fn_b = SCHEDULERS.get(scheduler_b, linear_schedule)

    # Phase A: structural (t=1 → crossover), n_a steps
    # Generate with shift=1 (we apply shift globally at the end)
    full_a = fn_a(n_a * 4, shift=1.0)  # oversample for better density
    phase_a = [t for t in full_a if t >= crossover]
    # If oversampled, subsample to n_a steps evenly
    if len(phase_a) > n_a:
        indices = [int(i * (len(phase_a) - 1) / (n_a - 1)) for i in range(n_a)]
        phase_a = [phase_a[idx] for idx in indices]
    elif len(phase_a) < n_a:
        # Fallback: linearly space between 1.0 and crossover
        phase_a = [1.0 - i * (1.0 - crossover) / n_a for i in range(n_a)]

    # Phase B: detail (crossover → 0), n_b steps
    full_b = fn_b(n_b * 4, shift=1.0)  # oversample
    phase_b = [t for t in full_b if t < crossover]
    if len(phase_b) > n_b:
        indices = [int(i * (len(phase_b) - 1) / (n_b - 1)) for i in range(n_b)]
        phase_b = [phase_b[idx] for idx in indices]
    elif len(phase_b) < n_b:
        # Fallback: linearly space between crossover and near-0
        phase_b = [crossover * (1.0 - (i + 1) / n_b) for i in range(n_b)]

    raw = phase_a + phase_b
    raw = sorted(raw, reverse=True)
    raw = [max(min(t, 1.0), 1e-6) for t in raw]
    return _apply_shift(raw, shift)


def beta57_schedule(num_steps: int, shift: float = 1.0) -> List[float]:
    """Beta distribution schedule with α=0.5, β=0.7 (aka "beta57").

    Popularised by the RES4LYF project for ComfyUI.  Uses the inverse CDF
    (percent-point function) of a Beta(0.5, 0.7) distribution to redistribute
    uniformly-spaced steps into a smooth S-curve that concentrates effort in
    the high-noise structural region while maintaining a moderate detail tail.

    Community consensus is that this schedule produces more coherent musical
    structure with ACE-Step than plain linear spacing.
    """
    return beta_schedule(num_steps, shift, alpha=0.5, beta_param=0.7)


def beta_schedule(
    num_steps: int,
    shift: float = 1.0,
    alpha: float = 0.5,
    beta_param: float = 0.7,
) -> List[float]:
    """Generalised beta distribution schedule.

    Maps N uniformly-spaced quantiles through the inverse CDF of
    Beta(alpha, beta_param).  The result is reversed so that timesteps
    descend from ~1.0 toward 0.

    Args:
        alpha:      Beta distribution α parameter (>0).  Lower values
                    push density toward the edges.
        beta_param: Beta distribution β parameter (>0).  When β < α,
                    density is shifted toward 1.0 (front-loaded).
    """
    from scipy.stats import beta as beta_dist

    raw = []
    for i in range(num_steps):
        # Uniform quantile in (0, 1), avoiding exact 0 and 1
        u = (i + 0.5) / num_steps
        # Inverse CDF maps uniform → beta-distributed
        t = 1.0 - beta_dist.ppf(u, alpha, beta_param)
        raw.append(t)

    # Ensure descending order and clamp
    raw = sorted(raw, reverse=True)
    raw = [max(min(t, 1.0), 1e-6) for t in raw]
    return _apply_shift(raw, shift)


def cosine_schedule(num_steps: int, shift: float = 1.0) -> List[float]:
    """Cosine annealing schedule.

    Uses a half-cosine curve to distribute steps, providing a smooth
    S-shaped density with gentle transitions at both extremes.  This
    is the schedule family used in DDPM/IDDPM research (Nichol & Dhariwal).

    Density is highest in the mid-range and tapers at the ends, giving
    balanced coverage of both structural and detail phases.
    """
    raw = []
    for i in range(num_steps):
        frac = i / num_steps
        # Cosine maps [0, 1] -> [1, 0] with S-shaped transition
        t = 0.5 * (1.0 + math.cos(math.pi * frac))
        raw.append(t)

    raw = [max(min(t, 1.0), 1e-6) for t in raw]
    return _apply_shift(raw, shift)


def power_schedule(
    num_steps: int,
    shift: float = 1.0,
    exponent: float = 2.0,
) -> List[float]:
    """Power-law schedule: t = (1 - i/N)^p.

    The exponent controls the curvature:
        p > 1:  Front-loaded — more steps for structure (default p=2).
        p = 1:  Identical to linear.
        p < 1:  Back-loaded — more steps for fine detail.

    A simple, interpretable schedule with a single tuning knob.
    """
    raw = []
    for i in range(num_steps):
        frac = i / num_steps  # 0 → nearly 1
        t = (1.0 - frac) ** exponent
        raw.append(t)

    raw = [max(min(t, 1.0), 1e-6) for t in raw]
    return _apply_shift(raw, shift)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCHEDULERS = {
    "linear": linear_schedule,
    "ddim_uniform": ddim_uniform_schedule,
    "sgm_uniform": sgm_uniform_schedule,
    "karras": sgm_uniform_schedule,      # alias — same Karras σ-ramp
    "bong_tangent": bong_tangent_schedule,
    "linear_quadratic": linear_quadratic_schedule,
    "beta57": beta57_schedule,
    "beta": beta_schedule,
    "cosine": cosine_schedule,
    "power": power_schedule,
    "composite": composite_schedule,
}

SCHEDULER_INFO = {
    "linear":           {"name": "Linear",               "description": "Uniform spacing (default)"},
    "ddim_uniform":     {"name": "DDIM Uniform",         "description": "Log-SNR uniform (S-shaped)"},
    "sgm_uniform":      {"name": "SGM-Uniform (Karras)", "description": "Karras σ-ramp (ρ=7). Front-loads structural steps."},
    "karras":           {"name": "SGM-Uniform (Karras)", "description": "Karras σ-ramp (ρ=7). Alias for SGM Uniform."},
    "bong_tangent":     {"name": "Tangent",              "description": "Front-loaded (structural focus)"},
    "linear_quadratic": {"name": "Linear-Quadratic",     "description": "Linear start, quadratic finish"},
    "beta57":           {"name": "Beta 57",              "description": "Beta(0.5, 0.7) — smooth S-curve, structural focus. From RES4LYF."},
    "beta":             {"name": "Beta",                 "description": "Configurable beta distribution schedule."},
    "cosine":           {"name": "Cosine",               "description": "Cosine annealing — balanced S-curve."},
    "power":            {"name": "Power",                "description": "Power-law t^p. p>1 = structural, p<1 = detail."},
    "composite":        {"name": "Composite",            "description": "Two-stage: different schedulers for structure vs detail"},
}

# Dynamic shift is not a scheduler, but a shift computation mode.
# It is invoked by callers when shift == -1 (auto mode).
# See compute_dynamic_shift() above.

VALID_SCHEDULERS = set(SCHEDULERS.keys())


def get_schedule(name: str, num_steps: int, shift: float = 1.0) -> List[float]:
    """Get a timestep schedule by name, supporting parameterized syntax.

    Simple schedulers::

        get_schedule("linear", 50, 5.0)
        get_schedule("bong_tangent", 50, 5.0)

    Parameterized schedulers — encode config in the name string::

        get_schedule("beta:0.5:0.7", 50, 5.0)        # beta with α=0.5, β=0.7
        get_schedule("power:2.0", 50, 5.0)            # power with exponent=2.0
        get_schedule("composite:bong_tangent+linear_quadratic:0.5:0.6", 50, 5.0)

    Composite format: ``composite:<scheduler_a>+<scheduler_b>:<crossover>:<split>``

    Returns:
        List of float timestep values, descending, in (0, 1].
    """
    name = name.lower().strip()

    # ── Composite scheduler parsed from string ───────────────────
    if name.startswith("composite"):
        # Parse: "composite:a+b:crossover:split"
        parts = name.split(":")
        sched_pair = parts[1] if len(parts) > 1 else "bong_tangent+linear"
        if "+" in sched_pair:
            a_name, b_name = sched_pair.split("+", 1)
        else:
            a_name, b_name = sched_pair, "linear"
        crossover = float(parts[2]) if len(parts) > 2 else 0.5
        split_frac = float(parts[3]) if len(parts) > 3 else 0.5
        return composite_schedule(
            num_steps, shift,
            scheduler_a=a_name.strip(),
            scheduler_b=b_name.strip(),
            crossover=crossover,
            split=split_frac,
        )

    # ── Parameterized beta schedule ──────────────────────────────
    if name.startswith("beta:"):
        # Parse: "beta:alpha:beta_param"
        parts = name.split(":")
        alpha = float(parts[1]) if len(parts) > 1 else 0.5
        beta_p = float(parts[2]) if len(parts) > 2 else 0.7
        return beta_schedule(num_steps, shift, alpha=alpha, beta_param=beta_p)

    # ── Parameterized power schedule ─────────────────────────────
    if name.startswith("power:"):
        # Parse: "power:exponent"
        parts = name.split(":")
        exponent = float(parts[1]) if len(parts) > 1 else 2.0
        return power_schedule(num_steps, shift, exponent=exponent)

    # ── Simple scheduler ─────────────────────────────────────────
    if name not in SCHEDULERS:
        valid = ", ".join(sorted(VALID_SCHEDULERS))
        raise ValueError(f"Unknown scheduler '{name}'. Valid schedulers: {valid}")
    return SCHEDULERS[name](num_steps, shift)


def get_scheduler(name: str):
    """Get a scheduler function by name (legacy API).

    For parameterized schedulers (composite, beta:α:β, power:exp),
    returns a wrapper that parses the name string.  For simple
    schedulers, returns the function directly.

    Returns:
        schedule_fn: Callable[[int, float], List[float]]

    Raises:
        ValueError if the scheduler name is not recognized.
    """
    clean = name.lower().strip()

    # Parameterized schedulers — return closures that delegate to get_schedule
    if clean.startswith(("composite", "beta:", "power:")):
        def _parameterized_wrapper(num_steps: int, shift: float = 1.0) -> List[float]:
            return get_schedule(name, num_steps, shift)
        return _parameterized_wrapper

    if clean not in SCHEDULERS:
        valid = ", ".join(sorted(VALID_SCHEDULERS))
        raise ValueError(f"Unknown scheduler '{clean}'. Valid schedulers: {valid}")
    return SCHEDULERS[clean]
