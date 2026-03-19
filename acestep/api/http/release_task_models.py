"""Pydantic request model definitions used by the `/release_task` flow."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from acestep.constants import DEFAULT_DIT_INSTRUCTION


class GenerateMusicRequest(BaseModel):
    """Typed request payload model for generation jobs.

    This schema mirrors the historical `api_server` request contract so legacy
    clients remain compatible while route handling is decomposed.
    """

    prompt: str = Field(default="", description="Text prompt describing the music")
    lyrics: str = Field(default="", description="Lyric text")

    # New API semantics:
    # - thinking=True: use 5Hz LM to generate audio codes (lm-dit behavior)
    # - thinking=False: do not use LM to generate codes (dit behavior)
    # Regardless of thinking, if some metas are missing, server may use LM to fill them.
    thinking: bool = False
    # Sample-mode requests auto-generate caption/lyrics/metas via LM (no user prompt).
    sample_mode: bool = False
    # Description for sample mode: auto-generate caption/lyrics from description query
    sample_query: str = Field(default="", description="Query/description for sample mode (use create_sample)")
    # Whether to use format_sample() to enhance input caption/lyrics
    use_format: bool = Field(default=False, description="Use format_sample() to enhance input (default: False)")
    # Model name for multi-model support (select which DiT model to use)
    model: Optional[str] = Field(default=None, description="Model name to use (e.g., 'acestep-v15-turbo')")

    bpm: Optional[int] = None
    key_scale: str = ""
    time_signature: str = ""
    vocal_language: str = "en"
    inference_steps: int = 8
    guidance_scale: float = 7.0
    use_random_seed: bool = True
    seed: Union[int, str] = -1

    reference_audio_path: Optional[str] = None
    src_audio_path: Optional[str] = None
    audio_duration: Optional[float] = None
    batch_size: Optional[int] = None

    repainting_start: float = 0.0
    repainting_end: Optional[float] = None

    instruction: str = DEFAULT_DIT_INSTRUCTION
    audio_cover_strength: float = 1.0
    cover_noise_strength: float = Field(
        default=0.0,
        description="Cover noise blending strength (0.0=pure noise, 1.0=closest to source audio). Used for cover/repaint tasks.",
    )
    # ── Custom fields (not in sdbds upstream) ──────────────────────
    tempo_scale: float = Field(default=1.0, description="Tempo scaling factor (0.5-2.0)")
    pitch_shift: int = Field(default=0, description="Pitch shift in semitones (-12 to +12)")
    enable_normalization: bool = Field(default=True, description="Enable loudness normalization")
    normalization_db: float = Field(default=-1.0, description="Target normalization loudness in dB")
    auto_master: bool = Field(default=True, description="Apply mastering profile to output audio")
    mastering_params: Optional[Dict] = Field(default=None, description="Override mastering parameters from console UI. e.g. {'mode': 'matchering', 'reference': '/path/to/ref.wav'}")
    latent_shift: float = Field(default=0.0, description="Latent space shift factor")
    latent_rescale: float = Field(default=1.0, description="Latent space rescale factor")
    vocoder_model: str = Field(default="", description="Optional HiFi-GAN vocoder model for enhanced decode (empty=default)")

    audio_code_string: str = Field(
        default="",
        description="User-provided audio semantic codes string for code-control generation. When non-empty, skips LM code generation.",
    )
    task_type: str = "text2music"
    analysis_only: bool = False
    full_analysis_only: bool = False

    use_adg: bool = False
    guidance_mode: str = Field(default="", description="Guidance mode override (e.g. 'cfg', 'pag')")
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0
    infer_method: str = "ode"  # "ode" or "sde" - diffusion inference method
    scheduler: str = Field(default="linear", description="Timestep scheduler: linear, ddim_uniform, sgm_uniform, bong_tangent, linear_quadratic")

    # ── PAG (Perturbed Attention Guidance) ─────────────────────────
    use_pag: bool = Field(default=False, description="Enable Perturbed Attention Guidance")
    pag_start: float = Field(default=0.30, description="PAG timestep start (0.0-1.0)")
    pag_end: float = Field(default=0.80, description="PAG timestep end (0.0-1.0)")
    pag_scale: float = Field(default=0.2, description="PAG guidance scale")

    # ── JKASS Fast solver parameters ──────────────────────────────
    beat_stability: float = Field(default=0.0, description="Momentum blending with previous step delta (0=off, 1=full)")
    frequency_damping: float = Field(default=0.0, description="Exponential decay on high-frequency bins (0=off)")
    temporal_smoothing: float = Field(default=0.0, description="Temporal kernel on velocity delta (0=off)")

    # ── Anti-Autotune ─────────────────────────────────────────────
    anti_autotune: float = Field(default=0.0, description="Spectral smoothing to reduce robotic/autotuned vocal artifacts (0=off, 1=full)")

    # ── Advanced Guidance Parameters ──────────────────────────────
    guidance_interval_decay: float = Field(default=0.0, description="Guidance interval decay rate")
    min_guidance_scale: float = Field(default=3.0, description="Minimum guidance scale for decayed guidance")
    guidance_scale_text: float = Field(default=0.0, description="Independent text prompt guidance (0=use main)")
    guidance_scale_lyric: float = Field(default=0.0, description="Independent lyric guidance (0=use main)")
    apg_momentum: float = Field(default=0.0, description="APG momentum (0=default internal)")
    apg_norm_threshold: float = Field(default=0.0, description="APG norm threshold (0=default 2.5)")
    omega_scale: float = Field(default=1.0, description="Omega guidance scale")
    erg_scale: float = Field(default=1.0, description="ERG guidance scale")

    # ── Iterative Refinement ──────────────────────────────────────
    refine_passes: int = Field(default=0, description="Number of refinement passes (0=disabled)")
    refine_strength: float = Field(default=0.3, description="Refinement noise strength")

    shift: float = Field(
        default=3.0,
        description="Timestep shift factor (range 1.0~5.0, default 3.0). Only effective for base models, not turbo models.",
    )
    timesteps: Optional[str] = Field(
        default=None,
        description="Custom timesteps (comma-separated, e.g., '0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0'). Overrides inference_steps and shift.",
    )

    audio_format: str = Field(
        default="mp3",
        description="Output audio format. Supported formats: 'flac', 'mp3', 'opus', 'aac', 'wav', 'wav32'. Default: 'mp3'",
    )
    use_tiled_decode: bool = True

    # 5Hz LM (server-side): used for metadata completion and (when thinking=True) codes generation.
    lm_model_path: Optional[str] = None  # e.g. "acestep-5Hz-lm-0.6B"
    lm_backend: Literal["vllm", "pt", "mlx", "custom-vllm"] = "vllm"

    constrained_decoding: bool = True
    constrained_decoding_debug: bool = False
    use_cot_caption: bool = True
    use_cot_language: bool = True
    is_format_caption: bool = False
    allow_lm_batch: bool = True
    track_name: Optional[str] = None
    track_classes: Optional[List[str]] = None

    lm_temperature: float = 0.85
    lm_cfg_scale: float = 2.5
    lm_top_k: Optional[int] = None
    lm_top_p: Optional[float] = 0.9
    lm_repetition_penalty: float = 1.0
    lm_negative_prompt: str = "NO USER INPUT"

    # ── Scoring & LRC output ──────────────────────────────────────
    get_lrc: bool = Field(default=False, description="Return LRC (timed lyrics) in response")
    get_scores: bool = Field(default=False, description="Compute quality scores for generated audio")
    score_scale: float = Field(default=0.1, description="Score computation scale factor")

    # ── Activation steering ───────────────────────────────────────
    steering_enabled: bool = Field(default=False, description="Enable TADA activation steering during generation")
    steering_loaded: List[str] = Field(default_factory=list, description="List of loaded steering concept names")
    steering_alphas: Dict[str, float] = Field(default_factory=dict, description="Per-concept alpha values")

    class Config:
        """Legacy pydantic config preserving prior population semantics."""

        allow_population_by_field_name = True
        allow_population_by_alias = True

