"""Checkpoint and model-loading helpers for service initialization."""

import gc
import importlib
import os
from typing import Optional

import torch
from loguru import logger

from acestep import gpu_config


class InitServiceLoaderMixin:
    """Helpers for heavy model component loading."""

    def _cuda_supports_bool_argsort(self) -> bool:
        """Return whether CUDA argsort supports bool tensors on the active device."""
        if not torch.cuda.is_available():
            return True
        target_device = str(getattr(self, "device", "cuda"))
        if not target_device.startswith("cuda"):
            target_device = "cuda"
        try:
            mask_cat = torch.tensor([[True, False]], device=target_device)
            _ = mask_cat.argsort(dim=1, descending=True, stable=True)
            return True
        except RuntimeError as exc:
            logger.debug(
                "[_cuda_supports_bool_argsort] Treating CUDA bool argsort probe failure as unsupported: {}",
                exc,
            )
            return False

    def _apply_cuda_bool_argsort_workaround(self) -> None:
        """Patch dynamic model helpers when bool argsort is unsupported on CUDA."""
        target_device = str(getattr(self, "device", ""))
        if not target_device.startswith("cuda"):
            return
        if self._cuda_supports_bool_argsort():
            return

        model_module_name = getattr(self.model.__class__, "__module__", "")
        if not model_module_name:
            return

        try:
            model_module = importlib.import_module(model_module_name)
        except Exception as exc:
            logger.warning(
                "[initialize_service] Failed to import model module for CUDA bool-argsort workaround: {}",
                exc,
            )
            return

        original_pack_sequences = getattr(model_module, "pack_sequences", None)
        if original_pack_sequences is None:
            return
        if getattr(original_pack_sequences, "__acestep_bool_argsort_patched__", False):
            return

        def _pack_sequences_cuda_compat(hidden1, hidden2, mask1, mask2):
            # ``pack_sequences`` only needs sortable integer-like masks here; keep
            # truthy/falsey semantics while avoiding CUDA bool argsort failures.
            if isinstance(mask1, torch.Tensor) and mask1.is_cuda and mask1.dtype == torch.bool:
                mask1 = mask1.to(torch.int32)
            if isinstance(mask2, torch.Tensor) and mask2.is_cuda and mask2.dtype == torch.bool:
                mask2 = mask2.to(torch.int32)
            return original_pack_sequences(hidden1, hidden2, mask1, mask2)

        _pack_sequences_cuda_compat.__acestep_bool_argsort_patched__ = True
        setattr(model_module, "pack_sequences", _pack_sequences_cuda_compat)
        logger.warning(
            "[initialize_service] Applied CUDA bool-argsort workaround to {}.pack_sequences",
            model_module_name,
        )

    def _load_main_model_from_checkpoint(
        self,
        *,
        model_checkpoint_path: str,
        device: str,
        use_flash_attention: bool,
        compile_model: bool,
        quantization: Optional[str],
    ) -> str:
        """Load DiT, apply compile/quantization options, and return selected attention backend."""
        from transformers import AutoModel

        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"ACE-Step V1.5 checkpoint not found at {model_checkpoint_path}")

        if torch.cuda.is_available():
            if getattr(self, "model", None) is not None:
                # Reset torch._dynamo BEFORE deleting the old model.
                # Compiled graph caches hold references to the model's tensors;
                # without this, del model won't actually free the CUDA memory.
                torch._dynamo.reset()
                del self.model
                self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except RuntimeError as exc:
                logger.warning(
                    "[initialize_service] cuda.synchronize() failed during pre-load cleanup: {}. "
                    "Continuing with fresh load attempt.",
                    exc,
                )

        if use_flash_attention and self.is_flash_attention_available(device):
            attn_implementation = "flash_attention_2"
        elif device == "cuda" and not gpu_config.cuda_supports_bfloat16():
            # Pre-Ampere GPUs (compute capability < 8.0) run in float16 which
            # can overflow in SDPA's fused softmax with longer sequences,
            # producing NaN/Inf latents.  Eager attention upcasts to float32
            # for softmax, avoiding the overflow.
            logger.info(
                "[initialize_service] Pre-Ampere CUDA detected: using eager "
                "attention for float16 numerical stability."
            )
            attn_implementation = "eager"
        else:
            if use_flash_attention:
                logger.warning(
                    f"[initialize_service] Flash attention requested but unavailable for device={device}. "
                    "Falling back to SDPA."
                )
            attn_implementation = "sdpa"

        attn_candidates = [attn_implementation]
        if "sdpa" not in attn_candidates:
            attn_candidates.append("sdpa")
        if "eager" not in attn_candidates:
            attn_candidates.append("eager")

        last_attn_error = None
        self.model = None
        for candidate in attn_candidates:
            try:
                logger.info(f"[initialize_service] Attempting to load model with attention implementation: {candidate}")
                self.model = AutoModel.from_pretrained(
                    model_checkpoint_path,
                    trust_remote_code=True,
                    attn_implementation=candidate,
                    dtype=self.dtype,
                )
                attn_implementation = candidate
                break
            except Exception as exc:
                last_attn_error = exc
                logger.warning(f"[initialize_service] Failed to load model with {candidate}: {exc}")

        if self.model is None:
            raise RuntimeError(
                f"Failed to load model with attention implementations {attn_candidates}: {last_attn_error}"
            ) from last_attn_error

        self.model.config._attn_implementation = attn_implementation
        self.config = self.model.config
        self._sync_alignment_config()
        self._apply_cuda_bool_argsort_workaround()

        # Merged models (SFT+Turbo weight averages) are NOT pure distillation
        # checkpoints — they can benefit from higher step counts, guidance, and
        # shift adjustments.  Override the inherited is_turbo flag.
        _model_basename = os.path.basename(model_checkpoint_path)
        if "merge" in _model_basename and getattr(self.config, "is_turbo", False):
            logger.info(
                "[_load_main_model_from_checkpoint] Merged model '{}' detected "
                "— overriding is_turbo=False",
                _model_basename,
            )
            self.config.is_turbo = False

        if not self.offload_to_cpu:
            self.model = self.model.to(device).to(self.dtype)
        elif not self.offload_dit_to_cpu:
            logger.info(f"[initialize_service] Keeping main model on {device} (persistent)")
            self.model = self.model.to(device).to(self.dtype)
        else:
            self.model = self.model.to("cpu").to(self.dtype)
        self.model.eval()

        # Replace checkpoint's vanilla generate_audio with our patched version
        # (solver/guidance registry, PAG support, etc.) — no disk files modified.
        from acestep.core.generation.monkeypatch_generate_audio import apply_generate_audio_monkeypatch
        apply_generate_audio_monkeypatch(self.model)

        if compile_model:
            self._ensure_len_for_compile(self.model, "model")
            self.model = torch.compile(self.model)

            if quantization is not None:
                from torchao.quantization import quantize_
                from torchao.quantization.quant_api import _is_linear
                if quantization == "int8_weight_only":
                    from torchao.quantization import Int8WeightOnlyConfig
                    quant_config = Int8WeightOnlyConfig()
                elif quantization == "int4_weight_only":
                    from torchao.quantization import Int4WeightOnlyConfig
                    quant_config = Int4WeightOnlyConfig(group_size=128, version=1)
                elif quantization == "fp8_weight_only":
                    from torchao.quantization import Float8WeightOnlyConfig
                    quant_config = Float8WeightOnlyConfig()
                elif quantization == "w8a8_dynamic":
                    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, MappingType
                    quant_config = Int8DynamicActivationInt8WeightConfig(act_mapping_type=MappingType.ASYMMETRIC)
                elif quantization == "nf4":
                    # NF4 uses a different path: manual per-layer weight conversion
                    # via torchao NF4Tensor (Normal Float 4-bit with non-uniform
                    # quantization levels tuned to neural network weight distributions).
                    from torchao.dtypes import to_nf4
                    import torch.nn as nn
                    nf4_count = 0
                    for name, module in self.model.named_modules():
                        if isinstance(module, nn.Linear):
                            skip = any(part in name.split(".") for part in ("tokenizer", "detokenizer"))
                            if not skip:
                                module.weight = nn.Parameter(
                                    to_nf4(module.weight), requires_grad=False
                                )
                                nf4_count += 1
                    logger.info(f"[initialize_service] NF4 applied to {nf4_count} linear layers")
                    quant_config = None  # skip quantize_() call below
                else:
                    raise ValueError(f"Unsupported quantization type: {quantization}")

                def _dit_filter_fn(module, fqn):
                    """Keep only DiT linear layers and exclude tokenizer/detokenizer paths."""
                    if not _is_linear(module, fqn):
                        return False
                    for part in fqn.split("."):
                        if part in ("tokenizer", "detokenizer"):
                            return False
                    return True

                if quant_config is not None:
                    quantize_(self.model, quant_config, filter_fn=_dit_filter_fn)
                logger.info(f"[initialize_service] DiT quantized with: {quantization}")

        silence_latent_path = os.path.join(model_checkpoint_path, "silence_latent.pt")
        if not os.path.exists(silence_latent_path):
            raise FileNotFoundError(f"Silence latent not found at {silence_latent_path}")
        self.silence_latent = torch.load(silence_latent_path, weights_only=True).transpose(1, 2)
        self.silence_latent = self.silence_latent.to(device).to(self.dtype)
        return attn_implementation

    def _load_vae_model(self, *, checkpoint_dir: str, device: str, compile_model: bool) -> str:
        """Load and optionally compile the VAE module."""
        from diffusers.models import AutoencoderOobleck

        vae_checkpoint_path = os.path.join(checkpoint_dir, "vae")
        if not os.path.exists(vae_checkpoint_path):
            raise FileNotFoundError(f"VAE checkpoint not found at {vae_checkpoint_path}")

        self.vae = AutoencoderOobleck.from_pretrained(vae_checkpoint_path)
        if not self.offload_to_cpu:
            vae_dtype = self._get_vae_dtype(device)
            self.vae = self.vae.to(device).to(vae_dtype)
        else:
            vae_dtype = self._get_vae_dtype("cpu")
            self.vae = self.vae.to("cpu").to(vae_dtype)
        self.vae.eval()

        if compile_model:
            self._ensure_len_for_compile(self.vae, "vae")
            self.vae = torch.compile(self.vae)

        return vae_checkpoint_path

    def _load_text_encoder_and_tokenizer(self, *, checkpoint_dir: str, device: str) -> str:
        """Load text tokenizer and embedding model."""
        from transformers import AutoModel, AutoTokenizer

        text_encoder_path = os.path.join(checkpoint_dir, "Qwen3-Embedding-0.6B")
        if not os.path.exists(text_encoder_path):
            raise FileNotFoundError(f"Text encoder not found at {text_encoder_path}")

        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_path)
        if not self.offload_to_cpu:
            self.text_encoder = self.text_encoder.to(device).to(self.dtype)
        else:
            self.text_encoder = self.text_encoder.to("cpu").to(self.dtype)
        self.text_encoder.eval()
        return text_encoder_path
