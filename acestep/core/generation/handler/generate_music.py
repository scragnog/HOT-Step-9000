"""Top-level ``generate_music`` orchestration mixin.

This module provides the public ``generate_music`` entry point extracted from
``AceStepHandler`` so orchestration stays separate from lower-level helpers.
"""

import traceback
from typing import Any, Dict, List, Optional, Union

import torch
from loguru import logger

from acestep.constants import DEFAULT_DIT_INSTRUCTION
from acestep.core.generation.spectral_smoothing import apply_spectral_smoothing
from acestep.gpu_config import (
    DIT_INFERENCE_VRAM_PER_BATCH,
    VRAM_SAFETY_MARGIN_GB,
    get_effective_free_vram_gb,
)


class GenerateMusicMixin:
    """Coordinate request prep, service execution, decode, and payload assembly.

    The host class is expected to implement helper methods invoked by this
    orchestration flow.
    """

    def _vram_preflight_check(
        self,
        actual_batch_size: int,
        audio_duration: Optional[float],
        guidance_scale: float,
    ) -> Optional[Dict[str, Any]]:
        """Check free VRAM headroom before attempting service_generate.

        Model weights are already resident in GPU memory at this point.  We
        only need to verify there is enough room for the diffusion-pass
        activations (intermediate attention maps, FFN buffers, noise tensors)
        plus a project-standard safety margin.

        Args:
            actual_batch_size: Number of samples being generated.
            audio_duration: Requested audio length in seconds, or None for default.
            guidance_scale: CFG guidance value; values > 1.0 indicate CFG is active
                and the DiT runs two forward passes per step (doubling activation memory).

        Returns:
            An error payload dict when VRAM is insufficient, or None when the
            check passes or no CUDA device is present (CPU/MPS/XPU fall through).
        """
        if not torch.cuda.is_available():
            return None

        if getattr(self, "offload_to_cpu", False):
            logger.debug(
                "[generate_music] VRAM pre-flight: skipping check "
                "(offload_to_cpu=True, models loaded one-at-a-time)"
            )
            return None

        duration_s = audio_duration or 60.0
        # CFG doubles forward-pass memory: two DiT evaluations per step.
        dit_key = "base" if guidance_scale > 1.0 else "turbo"
        per_batch_gb = DIT_INFERENCE_VRAM_PER_BATCH.get(dit_key, 0.6)
        # Longer audio = more latent frames (5 Hz rate) = more memory.
        duration_factor = max(1.0, duration_s / 60.0)
        needed_gb = per_batch_gb * actual_batch_size * duration_factor + VRAM_SAFETY_MARGIN_GB

        free_gb = get_effective_free_vram_gb()
        logger.info(
            "[generate_music] VRAM pre-flight: {:.2f} GB free, ~{:.2f} GB needed "
            "(batch={}, duration={:.0f}s, mode={}).",
            free_gb, needed_gb, actual_batch_size, duration_s, dit_key,
        )

        if free_gb >= needed_gb:
            return None

        msg = (
            f"Insufficient free VRAM: need ~{needed_gb:.1f} GB, "
            f"only {free_gb:.1f} GB available. "
            f"Reduce batch size (currently {actual_batch_size}) "
            f"or audio duration (currently {duration_s:.0f}s)."
        )
        logger.warning("[generate_music] VRAM pre-flight failed: {}", msg)
        return {
            "audios": [],
            "status_message": f"Error: {msg}",
            "extra_outputs": {},
            "success": False,
            "error": msg,
        }

    def generate_music(
        self,
        captions: str,
        lyrics: str,
        bpm: Optional[int] = None,
        key_scale: str = "",
        time_signature: str = "",
        vocal_language: str = "en",
        inference_steps: int = 8,
        guidance_scale: float = 7.0,
        use_random_seed: bool = True,
        seed: Optional[Union[str, float, int]] = -1,
        reference_audio=None,
        audio_duration: Optional[float] = None,
        batch_size: Optional[int] = None,
        src_audio=None,
        audio_code_string: Union[str, List[str]] = "",
        repainting_start: float = 0.0,
        repainting_end: Optional[float] = None,
        instruction: str = DEFAULT_DIT_INSTRUCTION,
        audio_cover_strength: float = 1.0,
        cover_noise_strength: float = 0.0,
        tempo_scale: float = 1.0,
        pitch_shift: int = 0,
        task_type: str = "text2music",
        use_adg: bool = False,
        guidance_mode: str = "",
        cfg_interval_start: float = 0.0,
        cfg_interval_end: float = 1.0,
        shift: float = 1.0,
        infer_method: str = "ode",
        use_tiled_decode: bool = True,
        timesteps: Optional[List[float]] = None,
        latent_shift: float = 0.0,
        latent_rescale: float = 1.0,
        # PAG (Perturbed-Attention Guidance) Parameters
        use_pag: bool = False,
        pag_start: float = 0.30,
        pag_end: float = 0.80,
        pag_scale: float = 0.2,
        # Steering Parameters
        steering_enabled: Optional[bool] = None,
        steering_loaded: Optional[List[str]] = None,
        steering_alphas: Optional[Dict[str, float]] = None,
        scheduler: str = "linear",
        # Anti-Autotune spectral smoothing (0=off, 1=full)
        anti_autotune: float = 0.0,
        # JKASS Fast solver parameters
        beat_stability: float = 0.0,
        frequency_damping: float = 0.0,
        temporal_smoothing: float = 0.0,
        # Advanced Guidance Parameters
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        apg_momentum: float = 0.0,
        apg_norm_threshold: float = 0.0,
        omega_scale: float = 1.0,
        erg_scale: float = 1.0,
        progress=None,
    ) -> Dict[str, Any]:
        """Generate audio from text/reference inputs and return response payload.

        Args:
            captions: Text prompt describing requested music.
            lyrics: Lyric text used for conditioning.
            reference_audio: Optional reference-audio payload.
            src_audio: Optional source audio for repaint/cover.
            inference_steps: Diffusion step count.
            guidance_scale: CFG guidance value.
            seed: Optional explicit seed from caller/UI.
            infer_method: Diffusion method name.
            timesteps: Optional custom timestep schedule.
            use_tiled_decode: Whether tiled VAE decode is used.
            latent_shift: Additive latent post-processing value.
            latent_rescale: Multiplicative latent post-processing value.
            progress: Optional callback taking ``(ratio, desc=...)``.

        Returns:
            Dict[str, Any]: Standard payload with generated audio tensors, status,
            intermediate outputs, success flag, and optional error text.

        Raises:
            No exceptions are re-raised. Runtime failures are converted into the
            returned error payload.
        """
        progress = self._resolve_generate_music_progress(progress)
        if self.model is None or self.vae is None or self.text_tokenizer is None or self.text_encoder is None:
            readiness_error = self._validate_generate_music_readiness()
            return readiness_error

        task_type, instruction = self._resolve_generate_music_task(
            task_type=task_type,
            audio_code_string=audio_code_string,
            instruction=instruction,
        )

        logger.info("[generate_music] Starting generation...")

        # --- Trigger word auto-injection ---
        # In advanced multi-adapter mode each slot may have its own trigger
        # word and tag_position.  Iterate all slots and compose them in order.
        # In basic mode fall back to the legacy _adapter_trigger_word attr.
        adapter_slots: dict = getattr(self, "_adapter_slots", {})
        if adapter_slots and self.lora_loaded:
            for _slot_id in sorted(adapter_slots.keys()):
                slot_info = adapter_slots[_slot_id]
                tw = slot_info.get("trigger_word", "")
                tp = slot_info.get("tag_position", "prepend") or "prepend"
                if not tw:
                    continue
                current = captions.strip() if captions else ""
                if tw.lower() in current.lower():
                    logger.info(f"[generate_music] Slot {_slot_id} trigger word '{tw}' already present, skipping")
                    continue
                if tp == "replace":
                    captions = tw
                    logger.info(f"[generate_music] Slot {_slot_id} trigger word replaced caption → '{tw}'")
                elif tp == "append":
                    captions = f"{current}, {tw}" if current else tw
                    logger.info(f"[generate_music] Slot {_slot_id} trigger word appended → '{captions}'")
                else:  # prepend (default)
                    captions = f"{tw}, {current}" if current else tw
                    logger.info(f"[generate_music] Slot {_slot_id} trigger word prepended → '{captions}'")
        else:
            # Basic mode: single _adapter_trigger_word
            trigger_word = getattr(self, "_adapter_trigger_word", "")
            tag_position = getattr(self, "_adapter_tag_position", "")
            if trigger_word and self.lora_loaded:
                caption_stripped = captions.strip() if captions else ""
                if trigger_word.lower() not in caption_stripped.lower():
                    if tag_position == "replace":
                        captions = trigger_word
                        logger.info(f"[generate_music] Trigger word replaced caption → '{trigger_word}'")
                    elif tag_position == "append":
                        captions = f"{caption_stripped}, {trigger_word}" if caption_stripped else trigger_word
                        logger.info(f"[generate_music] Trigger word appended → '{captions}'")
                    else:  # prepend (default)
                        captions = f"{trigger_word}, {caption_stripped}" if caption_stripped else trigger_word
                        logger.info(f"[generate_music] Trigger word prepended → '{captions}'")

        if progress:
            progress(0.51, desc="Preparing inputs...")
        logger.info("[generate_music] Preparing inputs...")

        runtime = self._prepare_generate_music_runtime(
            batch_size=batch_size,
            audio_duration=audio_duration,
            repainting_end=repainting_end,
            seed=seed,
            use_random_seed=use_random_seed,
        )
        actual_batch_size = runtime["actual_batch_size"]
        actual_seed_list = runtime["actual_seed_list"]
        seed_value_for_ui = runtime["seed_value_for_ui"]
        audio_duration = runtime["audio_duration"]
        repainting_end = runtime["repainting_end"]

        try:
            # Temporary override of steering configuration if present in GenerationParams
            prev_steering_enabled = getattr(self, "steering_enabled", False)
            prev_steering_config = getattr(self, "steering_config", {})
            
            has_req_steering = steering_enabled is not None
            if has_req_steering:
                self.steering_enabled = steering_enabled
                temp_config = {}
                for concept in (steering_loaded or []):
                    alpha = (steering_alphas or {}).get(concept, 1.0)
                    temp_config[concept] = {
                        "alpha": alpha,
                        "layers": "tf7",
                        "mode": "cond_only",
                    }
                self.steering_config = temp_config

            refer_audios, processed_src_audio, audio_error = self._prepare_reference_and_source_audio(
                reference_audio=reference_audio,
                src_audio=src_audio,
                audio_code_string=audio_code_string,
                actual_batch_size=actual_batch_size,
                task_type=task_type,
                tempo_scale=tempo_scale,
                pitch_shift=pitch_shift,
            )
            if audio_error is not None:
                return audio_error

            service_inputs = self._prepare_generate_music_service_inputs(
                actual_batch_size=actual_batch_size,
                processed_src_audio=processed_src_audio,
                audio_duration=audio_duration,
                captions=captions,
                lyrics=lyrics,
                vocal_language=vocal_language,
                instruction=instruction,
                bpm=bpm,
                key_scale=key_scale,
                time_signature=time_signature,
                task_type=task_type,
                audio_code_string=audio_code_string,
                repainting_start=repainting_start,
                repainting_end=repainting_end,
            )
            vram_error = self._vram_preflight_check(
                actual_batch_size=actual_batch_size,
                audio_duration=audio_duration,
                guidance_scale=guidance_scale,
            )
            if vram_error is not None:
                return vram_error

            service_run = self._run_generate_music_service_with_progress(
                progress=progress,
                actual_batch_size=actual_batch_size,
                audio_duration=audio_duration,
                inference_steps=inference_steps,
                timesteps=timesteps,
                service_inputs=service_inputs,
                refer_audios=refer_audios,
                guidance_scale=guidance_scale,
                actual_seed_list=actual_seed_list,
                audio_cover_strength=audio_cover_strength,
                cover_noise_strength=cover_noise_strength,
                guidance_mode=guidance_mode if guidance_mode else ("adg" if use_adg else "apg"),
                cfg_interval_start=cfg_interval_start,
                cfg_interval_end=cfg_interval_end,
                shift=shift,
                infer_method=infer_method,
                use_pag=use_pag,
                pag_start=pag_start,
                pag_end=pag_end,
                pag_scale=pag_scale,
                scheduler=scheduler,
                guidance_scale_text=guidance_scale_text,
                guidance_scale_lyric=guidance_scale_lyric,
                apg_momentum=apg_momentum,
                apg_norm_threshold=apg_norm_threshold,
                omega_scale=omega_scale,
                erg_scale=erg_scale,
                beat_stability=beat_stability,
                frequency_damping=frequency_damping,
                temporal_smoothing=temporal_smoothing,
            )
            outputs = service_run["outputs"]
            infer_steps_for_progress = service_run["infer_steps_for_progress"]

            pred_latents, time_costs = self._prepare_generate_music_decode_state(
                outputs=outputs,
                infer_steps_for_progress=infer_steps_for_progress,
                actual_batch_size=actual_batch_size,
                audio_duration=audio_duration,
                latent_shift=latent_shift,
                latent_rescale=latent_rescale,
            )
            # Apply anti-autotune spectral smoothing before VAE decode
            if anti_autotune > 0:
                logger.info(f"[generate_music] Applying anti-autotune spectral smoothing (strength={anti_autotune:.2f})")
                pred_latents = apply_spectral_smoothing(pred_latents, anti_autotune)
            pred_wavs, pred_latents_cpu, time_costs = self._decode_generate_music_pred_latents(
                pred_latents=pred_latents,
                progress=progress,
                use_tiled_decode=use_tiled_decode,
                time_costs=time_costs,
            )
            return self._build_generate_music_success_payload(
                outputs=outputs,
                pred_wavs=pred_wavs,
                pred_latents_cpu=pred_latents_cpu,
                time_costs=time_costs,
                seed_value_for_ui=seed_value_for_ui,
                actual_batch_size=actual_batch_size,
                progress=progress,
            )

        except Exception as exc:
            error_msg = f"Error: {exc!s}\n{traceback.format_exc()}"
            logger.exception("[generate_music] Generation failed")
            return {
                "audios": [],
                "status_message": error_msg,
                "extra_outputs": {},
                "success": False,
                "error": f"{exc!s}",
            }
        
        finally:
            if has_req_steering:
                self.steering_enabled = prev_steering_enabled
                self.steering_config = prev_steering_config
