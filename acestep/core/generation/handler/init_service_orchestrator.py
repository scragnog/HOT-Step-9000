"""Top-level initialization orchestration for the handler."""

import gc
import os
import traceback
from pathlib import Path
from typing import Optional, Tuple

import torch
from loguru import logger

from acestep import gpu_config

_ROCM_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _log_vram(label: str) -> None:
    """Log current CUDA VRAM usage with a contextual label."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    logger.info(f"[VRAM] {label}: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB")


def _resolve_rocm_dtype() -> torch.dtype:
    """Return a safe model dtype for ROCm/HIP devices.

    Uses ``float32`` by default to avoid segfaults from incomplete
    ``bfloat16`` kernel support on some ROCm GPU configurations (e.g.
    AMD iGPUs on Strix Halo).  Set the ``ACESTEP_ROCM_DTYPE`` environment
    variable to ``float16`` or ``bfloat16`` to override for hardware that
    fully supports those formats.
    """
    raw = os.environ.get("ACESTEP_ROCM_DTYPE", "float32").strip().lower()
    dtype = _ROCM_DTYPE_MAP.get(raw)
    if dtype is None:
        logger.warning(
            f"[initialize_service] Unknown ACESTEP_ROCM_DTYPE={raw!r}; "
            "falling back to float32."
        )
        dtype = torch.float32
    return dtype


class InitServiceOrchestratorMixin:
    """Public ``initialize_service`` orchestration entrypoint."""

    def _reset_lora_state(self) -> None:
        """Reset all LoRA/LoKr state to defaults.

        Must be called when the underlying model is replaced (switch or reinit)
        to prevent stale _base_decoder from causing dimension mismatches.
        """
        self._base_decoder = None
        self.lora_loaded = False
        self.use_lora = False
        self.lora_scale = 1.0
        self._active_loras = {}
        self._lora_adapter_registry = {}
        self._lora_active_adapter = None
        if hasattr(self, "_adapter_type"):
            self._adapter_type = None
        if hasattr(self, "_lora_scale_state"):
            self._lora_scale_state = {}
        # Reset advanced adapter slot state (weight-space merging system)
        self._adapter_slots = {}
        self._next_slot_id = 0
        self._merged_dirty = False
        # Reset LoraService registry if present
        lora_svc = getattr(self, "_lora_service", None)
        if lora_svc is not None:
            lora_svc.registry = {}
            lora_svc.scale_state = {}
            lora_svc.active_adapter = None
            lora_svc.last_scale_report = {}
        # Reset Redmond Mode delta/raw_base (keep adapter_path for re-apply)
        from acestep.core.generation.handler.redmond_mode import reset_redmond_state
        reset_redmond_state(self)
        logger.info("[_reset_lora_state] LoRA/LoKr/adapter-slot state cleared")

    # Sentinel value to distinguish "caller didn't provide quantization"
    # from "caller wants quantization=None" (i.e., full precision).
    _QUANT_UNCHANGED = object()

    def switch_dit_model(
        self,
        config_path: str,
        use_flash_attention: bool = False,
        quantization: object = _QUANT_UNCHANGED,
    ) -> Tuple[str, bool]:
        """Hot-swap the DiT model checkpoint and/or quantization without reloading VAE/text encoder.

        Cleans up LoRA state and resets _base_decoder to prevent stale backups.

        Args:
            config_path: Model directory name under checkpoints/ (e.g. 'acestep-v15-turbo').
            use_flash_attention: Whether to request flash attention for the new model.
            quantization: Optional quantization mode ('int8_weight_only', 'nf4', etc.) or
                None to disable quantization.  Uses sentinel ``_QUANT_UNCHANGED`` when not
                provided so the caller can explicitly pass ``None`` to mean "full precision".

        Returns:
            Tuple of (status_message, success_bool).
        """
        try:
            # Capture Redmond state BEFORE reset clears it
            _redmond_was_enabled = getattr(self, "_redmond_enabled", False)
            _redmond_path = getattr(self, "_redmond_adapter_path", "")
            _redmond_scale = getattr(self, "_redmond_scale", 0.7)

            _log_vram("switch_dit_model start")

            # Clean up LoRA state from previous model
            if self.lora_loaded:
                try:
                    msg = self.unload_lora()
                    logger.info(f"[switch_dit_model] Unloaded LoRA before switch: {msg}")
                except Exception as exc:
                    logger.warning(f"[switch_dit_model] Failed to unload LoRA: {exc}")
            self._reset_lora_state()
            _log_vram("after LoRA unload + reset")

            # Resolve checkpoint path
            base_root = (self.last_init_params or {}).get("project_root") or self._get_project_root()
            checkpoint_dir = os.path.join(base_root, "checkpoints")
            model_path = os.path.join(checkpoint_dir, config_path)

            if not os.path.exists(model_path):
                return f"Model checkpoint not found: {model_path}", False

            # Reuse compile/quantization settings from last init
            params = self.last_init_params or {}
            compile_model = params.get("compile_model", False)

            # Handle quantization change
            if quantization is not self._QUANT_UNCHANGED:
                effective_quant = quantization  # may be None (= full precision) or a string
                # Validate torchao availability when enabling quantization
                if effective_quant is not None:
                    try:
                        import torchao  # noqa: F401
                    except ImportError:
                        return (
                            "torchao is required for quantization but is not installed. "
                            "Please install torchao to use quantization features.",
                            False,
                        )
                    # Auto-enable compile_model — quantization requires it
                    if not compile_model:
                        logger.info(
                            "[switch_dit_model] Auto-enabling compile_model "
                            "(required for quantization)"
                        )
                        compile_model = True
                        if self.last_init_params is not None:
                            self.last_init_params["compile_model"] = True
                # Update handler state
                self.quantization = effective_quant
                if self.last_init_params is not None:
                    self.last_init_params["quantization"] = effective_quant
                logger.info(f"[switch_dit_model] Quantization changed to: {effective_quant}")
            else:
                effective_quant = params.get("quantization", None)

            self._sync_model_code_if_needed(config_path, Path(checkpoint_dir))

            self._load_main_model_from_checkpoint(
                model_checkpoint_path=model_path,
                device=str(self.device),
                use_flash_attention=use_flash_attention,
                compile_model=compile_model,
                quantization=effective_quant,
            )
            _log_vram("after model load")

            # Update last_init_params to reflect new config
            if self.last_init_params is not None:
                self.last_init_params["config_path"] = config_path
                self.last_init_params["use_flash_attention"] = use_flash_attention

            attn = getattr(self.config, "_attn_implementation", "eager")
            quant_label = effective_quant or "none"
            status = f"[OK] Switched to {config_path} on {self.device} (attn={attn}, quant={quant_label})"
            logger.info(f"[switch_dit_model] {status}")

            # Re-apply Redmond Mode if it was active before the switch (uses pre-reset values)
            # Redmond is incompatible with quantized weights (AffineQuantizedTensor dispatch)
            if _redmond_was_enabled and _redmond_path and self.quantization is None:
                if os.path.isdir(_redmond_path):
                    logger.info(f"[switch_dit_model] Re-applying Redmond Mode at scale {_redmond_scale:.2f}")
                    try:
                        from acestep.core.generation.handler.redmond_mode import apply_redmond_at_startup
                        apply_redmond_at_startup(self, _redmond_path, _redmond_scale)
                    except Exception as exc:
                        logger.warning(f"[switch_dit_model] Failed to re-apply Redmond: {exc}")
            elif _redmond_was_enabled and self.quantization is not None:
                logger.info(
                    "[switch_dit_model] Redmond Mode was active but quantization is enabled — "
                    "skipping Redmond re-application (incompatible with quantized weights)"
                )

            # Post-switch VRAM cleanup: release cached allocator blocks
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("[switch_dit_model] Post-switch VRAM cleanup complete")
                _log_vram("switch_dit_model complete")

            return status, True
        except Exception as exc:
            error_msg = f"Failed to switch model to {config_path}: {exc}"
            logger.exception(f"[switch_dit_model] {error_msg}")
            return error_msg, False

    def initialize_service(
        self,
        project_root: str,
        config_path: str,
        device: str = "auto",
        use_flash_attention: bool = False,
        compile_model: bool = False,
        offload_to_cpu: bool = False,
        offload_dit_to_cpu: bool = False,
        quantization: Optional[str] = None,
        prefer_source: Optional[str] = None,
        use_mlx_dit: bool = True,
        vae_model: str = "stock",
    ) -> Tuple[str, bool]:
        """Initialize model artifacts and runtime backends for generation.

        This method intentionally supports repeated calls to reinitialize models
        with new settings; it does not short-circuit when components are already loaded.
        """
        try:
            # Clean up stale LoRA state from any previous model before reinit
            self._reset_lora_state()

            if config_path is None:
                config_path = "acestep-v15-turbo"
                logger.warning(
                    "[initialize_service] config_path not set; defaulting to 'acestep-v15-turbo'."
                )

            resolved_device = self._resolve_initialize_device(device)
            self.device = resolved_device
            self.offload_to_cpu = offload_to_cpu
            self.offload_dit_to_cpu = offload_dit_to_cpu

            normalized_compile, normalized_quantization, mlx_compile_requested = self._configure_initialize_runtime(
                device=resolved_device,
                compile_model=compile_model,
                quantization=quantization,
            )
            self.compiled = normalized_compile
            if resolved_device == "cuda" and gpu_config.is_rocm_available():
                self.dtype = _resolve_rocm_dtype()
                logger.info(
                    f"[initialize_service] ROCm/HIP device detected: using dtype={self.dtype} "
                    "(set ACESTEP_ROCM_DTYPE=bfloat16 or float16 to override)"
                )
            else:
                self.dtype = torch.bfloat16 if resolved_device in ["cuda", "xpu"] else torch.float32
            self.quantization = normalized_quantization
            self._validate_quantization_setup(
                quantization=self.quantization,
                compile_model=normalized_compile,
            )

            base_root = project_root or self._get_project_root()
            checkpoint_dir = os.path.join(base_root, "checkpoints")
            checkpoint_path = Path(checkpoint_dir)

            precheck_failure = self._ensure_models_present(
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                prefer_source=prefer_source,
            )
            if precheck_failure is not None:
                self.model = None
                self.vae = None
                self.text_encoder = None
                self.text_tokenizer = None
                self.config = None
                self.silence_latent = None
                return precheck_failure

            self._sync_model_code_if_needed(config_path, checkpoint_path)

            model_path = os.path.join(checkpoint_dir, config_path)
            self._load_main_model_from_checkpoint(
                model_checkpoint_path=model_path,
                device=resolved_device,
                use_flash_attention=use_flash_attention,
                compile_model=normalized_compile,
                quantization=self.quantization,
            )
            vae_path = self._load_vae_model(
                checkpoint_dir=checkpoint_dir,
                device=resolved_device,
                compile_model=normalized_compile,
                vae_model=vae_model,
            )
            text_encoder_path = self._load_text_encoder_and_tokenizer(
                checkpoint_dir=checkpoint_dir,
                device=resolved_device,
            )

            mlx_dit_status, mlx_vae_status = self._initialize_mlx_backends(
                device=resolved_device,
                use_mlx_dit=use_mlx_dit,
                mlx_compile_requested=mlx_compile_requested,
            )

            status_msg = self._build_initialize_status_message(
                device=resolved_device,
                model_path=model_path,
                vae_path=vae_path,
                text_encoder_path=text_encoder_path,
                dtype=self.dtype,
                attention=getattr(self.config, "_attn_implementation", "eager"),
                compile_model=normalized_compile,
                mlx_compile_requested=mlx_compile_requested,
                offload_to_cpu=offload_to_cpu,
                offload_dit_to_cpu=offload_dit_to_cpu,
                mlx_dit_status=mlx_dit_status,
                mlx_vae_status=mlx_vae_status,
            )

            self.last_init_params = {
                "project_root": project_root,
                "config_path": config_path,
                "device": resolved_device,
                "use_flash_attention": use_flash_attention,
                "compile_model": normalized_compile,
                "offload_to_cpu": offload_to_cpu,
                "offload_dit_to_cpu": offload_dit_to_cpu,
                "quantization": self.quantization,
                "use_mlx_dit": use_mlx_dit,
                "prefer_source": prefer_source,
                "vae_model": vae_model,
            }

            return status_msg, True
        except Exception as exc:
            self.model = None
            self.vae = None
            self.text_encoder = None
            self.text_tokenizer = None
            self.config = None
            self.silence_latent = None
            error_msg = f"Error initializing model: {str(exc)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.exception(error_msg)
            return error_msg, False

