"""
Business Logic Handler
Encapsulates all data processing and business logic as a bridge between model and UI
"""
import os
import sys

# ---- File-based generation log for diagnostics ----
# All loguru output is written to the session's python_api.log (or fallback to generation.log)
_default_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
_session_log_dir = os.environ.get("ACESTEP_LOG_DIR")

if _session_log_dir:
    os.makedirs(_session_log_dir, exist_ok=True)
    _log_file_path = os.path.join(_session_log_dir, "python_api.log")
else:
    os.makedirs(_default_log_dir, exist_ok=True)
    _log_file_path = os.path.join(_default_log_dir, "generation.log")

from loguru import logger as _early_logger
_early_logger.add(
    _log_file_path,
    rotation="10 MB",
    retention=3,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}",
    level="DEBUG",
)

# Disable tokenizers parallelism to avoid fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import threading
from typing import Optional

import torch
import warnings

from acestep.core.generation.handler import (
    AudioCodesMixin,
    BatchPrepMixin,
    ConditioningBatchMixin,
    ConditioningEmbedMixin,
    ConditioningMaskMixin,
    ConditioningTargetMixin,
    ConditioningTextMixin,
    DiffusionMixin,
    GenerateMusicDecodeMixin,
    GenerateMusicExecuteMixin,
    GenerateMusicMixin,
    GenerateMusicPayloadMixin,
    GenerateMusicRequestMixin,
    InitServiceMixin,
    IoAudioMixin,
    LyricScoreMixin,
    LyricTimestampMixin,
    LoraManagerMixin,
    MemoryUtilsMixin,
    MetadataMixin,
    MlxDitInitMixin,
    MlxVaeDecodeNativeMixin,
    MlxVaeEncodeNativeMixin,
    MlxVaeInitMixin,
    PaddingMixin,
    ProgressMixin,
    PromptMixin,
    ServiceGenerateMixin,
    TrainingPresetMixin,
    TaskUtilsMixin,
    VaeDecodeChunksMixin,
    VaeDecodeMixin,
    VaeEncodeChunksMixin,
    VaeEncodeMixin,
    ServiceGenerateRequestMixin,
    ServiceGenerateExecuteMixin,
    ServiceGenerateOutputsMixin,
    SteeringMixin,
)
from acestep.core.generation.handler.lora.advanced_adapter_mixin import AdvancedAdapterMixin
from acestep.gpu_config import get_gpu_memory_gb, get_global_gpu_config, get_effective_free_vram_gb


warnings.filterwarnings("ignore")


class AceStepHandler(
    DiffusionMixin,
    GenerateMusicMixin,
    GenerateMusicDecodeMixin,
    GenerateMusicPayloadMixin,
    GenerateMusicExecuteMixin,
    GenerateMusicRequestMixin,
    AudioCodesMixin,
    BatchPrepMixin,
    ConditioningBatchMixin,
    ConditioningEmbedMixin,
    ConditioningMaskMixin,
    ConditioningTargetMixin,
    ConditioningTextMixin,
    IoAudioMixin,
    InitServiceMixin,
    LyricScoreMixin,
    LyricTimestampMixin,
    AdvancedAdapterMixin,
    LoraManagerMixin,
    MemoryUtilsMixin,
    MetadataMixin,
    MlxDitInitMixin,
    MlxVaeDecodeNativeMixin,
    MlxVaeEncodeNativeMixin,
    MlxVaeInitMixin,
    PaddingMixin,
    ProgressMixin,
    PromptMixin,
    ServiceGenerateMixin,
    TrainingPresetMixin,
    TaskUtilsMixin,
    VaeDecodeChunksMixin,
    VaeDecodeMixin,
    VaeEncodeChunksMixin,
    VaeEncodeMixin,
    ServiceGenerateRequestMixin,
    ServiceGenerateExecuteMixin,
    ServiceGenerateOutputsMixin,
    SteeringMixin,
):
    """ACE-Step Business Logic Handler"""
    
    def __init__(self):
        """Initialize runtime model handles, feature flags, and generation state."""
        self.model = None
        self.config = None
        self.device = "cpu"
        self.dtype = torch.float32  # Will be set based on device in initialize_service

        # VAE for audio encoding/decoding
        self.vae = None
        
        # Text encoder and tokenizer
        self.text_encoder = None
        self.text_tokenizer = None
        
        # Silence latent for initialization
        self.silence_latent = None
        
        # Sample rate
        self.sample_rate = 48000
        
        # Reward model (temporarily disabled)
        self.reward_model = None
        
        # Batch size
        self.batch_size = 2
        
        # Custom layers config
        self.custom_layers_config = {2: [6], 3: [10, 11], 4: [3], 5: [8, 9], 6: [8]}
        self.offload_to_cpu = False
        self.offload_dit_to_cpu = False
        self.compiled = False
        self.current_offload_cost = 0.0
        self.disable_tqdm = os.environ.get("ACESTEP_DISABLE_TQDM", "").lower() in ("1", "true", "yes") or not getattr(sys.stderr, 'isatty', lambda: False)()
        self.debug_stats = os.environ.get("ACESTEP_DEBUG_STATS", "").lower() in ("1", "true", "yes")
        self._last_diffusion_per_step_sec: Optional[float] = None
        self._progress_estimates_lock = threading.Lock()
        self._progress_estimates = {"records": []}
        self._progress_estimates_path = os.path.join(
            self._get_project_root(),
            ".cache",
            "acestep",
            "progress_estimates.json",
        )
        self._load_progress_estimates()
        self.last_init_params = None
        
        # Quantization state - tracks if model is quantized (int8_weight_only, fp8_weight_only, or w8a8_dynamic)
        # Populated during initialize_service, remains None if quantization is disabled
        self.quantization = None
        
        # LoRA state
        self.lora_loaded = False
        self.use_lora = False
        self.lora_scale = 1.0  # LoRA influence scale (0-1), mirrors active adapter's scale
        self._base_decoder = None  # Backup of original decoder state_dict (CPU) for memory efficiency
        self._active_loras = {}  # adapter_name -> scale (per-adapter)
        self._lora_adapter_registry = {}  # adapter_name -> explicit scaling targets
        self._lora_active_adapter = None

        # Advanced adapter state (slot-based weight-space merging)
        self._adapter_slots = {}      # slot_id -> {path, name, type, delta, scale, group_scales}
        self._next_slot_id = 0
        self._merged_dirty = False
        self.lora_group_scales = {"self_attn": 1.0, "cross_attn": 1.0, "mlp": 1.0, "cond_embed": 1.0}

        # Activation steering (TADA)
        self.steering_enabled = False
        self.steering_vectors = {}      # concept_name -> loaded vector dict
        self.steering_config = {}       # concept_name -> {"alpha": float, "layers": str, "mode": str}

        # MLX DiT acceleration (macOS Apple Silicon only)
        self.mlx_decoder = None
        self.use_mlx_dit = False
        self.mlx_dit_compiled = False

        # MLX VAE acceleration (macOS Apple Silicon only)
        self.mlx_vae = None
        self.use_mlx_vae = False

