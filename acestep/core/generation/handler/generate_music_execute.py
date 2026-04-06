"""Execution helper for ``generate_music`` service invocation with progress tracking."""

from typing import Any, Dict, List, Optional, Sequence


class GenerateMusicExecuteMixin:
    """Run service generation under diffusion progress estimation lifecycle."""

    def _run_generate_music_service_with_progress(
        self,
        progress: Any,
        actual_batch_size: int,
        audio_duration: Optional[float],
        inference_steps: int,
        timesteps: Optional[Sequence[float]],
        service_inputs: Dict[str, Any],
        refer_audios: Optional[List[Any]],
        guidance_scale: float,
        actual_seed_list: Optional[List[int]],
        audio_cover_strength: float,
        cover_noise_strength: float,
        guidance_mode: str,
        cfg_interval_start: float,
        cfg_interval_end: float,
        shift: float,
        infer_method: str,
        # PAG (Perturbed-Attention Guidance) Parameters
        use_pag: bool = False,
        pag_start: float = 0.30,
        pag_end: float = 0.80,
        pag_scale: float = 0.2,
        scheduler: str = "linear",
        # Advanced guidance parameters
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        apg_momentum: float = 0.0,
        apg_norm_threshold: float = 2.5,
        omega_scale: float = 1.0,
        erg_scale: float = 1.0,
        # JKASS Fast solver parameters
        beat_stability: float = 0.0,
        frequency_damping: float = 0.0,
        temporal_smoothing: float = 0.0,
        # STORK solver parameters
        stork_substeps: int = 50,
    ) -> Dict[str, Any]:
        """Invoke ``service_generate`` with real-time step progress from the DiT loop."""
        infer_steps_for_progress = len(timesteps) if timesteps else inference_steps
        progress(0.52, desc="Generating music...")

        outputs = self.service_generate(
            captions=service_inputs["captions_batch"],
            lyrics=service_inputs["lyrics_batch"],
            metas=service_inputs["metas_batch"],
            vocal_languages=service_inputs["vocal_languages_batch"],
            refer_audios=refer_audios,
            target_wavs=service_inputs["target_wavs_tensor"],
            infer_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=actual_seed_list,
            repainting_start=service_inputs["repainting_start_batch"],
            repainting_end=service_inputs["repainting_end_batch"],
            instructions=service_inputs["instructions_batch"],
            audio_cover_strength=audio_cover_strength,
            cover_noise_strength=cover_noise_strength,
            use_adg=False,  # Legacy compat
            guidance_mode=guidance_mode,
            cfg_interval_start=cfg_interval_start,
            cfg_interval_end=cfg_interval_end,
            shift=shift,
            infer_method=infer_method,
            audio_code_hints=service_inputs["audio_code_hints_batch"],
            return_intermediate=service_inputs["should_return_intermediate"],
            timesteps=timesteps,
            use_pag=use_pag,
            pag_start=pag_start,
            pag_end=pag_end,
            pag_scale=pag_scale,
            scheduler=scheduler,
            progress=progress,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            apg_momentum=apg_momentum,
            apg_norm_threshold=apg_norm_threshold,
            omega_scale=omega_scale,
            erg_scale=erg_scale,
            beat_stability=beat_stability,
            frequency_damping=frequency_damping,
            temporal_smoothing=temporal_smoothing,
            stork_substeps=stork_substeps,
        )
        return {"outputs": outputs, "infer_steps_for_progress": infer_steps_for_progress}

