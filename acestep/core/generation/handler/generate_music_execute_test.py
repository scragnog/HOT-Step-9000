"""Unit tests for ``generate_music`` execution helper mixin."""

import unittest

from acestep.core.generation.handler.generate_music_execute import GenerateMusicExecuteMixin


class _Host(GenerateMusicExecuteMixin):
    """Minimal host implementing progress/service stubs for execute helper tests."""

    def __init__(self):
        """Capture calls for assertions."""
        self.service_calls = 0
        self.service_kwargs = {}

    def service_generate(self, **kwargs):
        """Record service invocation and return minimal output payload."""
        self.service_calls += 1
        self.service_kwargs = kwargs
        return {"target_latents": "ok"}


class GenerateMusicExecuteMixinTests(unittest.TestCase):
    """Verify progress lifecycle and service forwarding behavior."""

    def test_run_service_with_progress_invokes_service(self):
        """Helper should call service once and pass progress through."""
        host = _Host()
        out = host._run_generate_music_service_with_progress(
            progress=lambda *args, **kwargs: None,
            actual_batch_size=1,
            audio_duration=10.0,
            inference_steps=8,
            timesteps=None,
            service_inputs={
                "captions_batch": ["c"],
                "lyrics_batch": ["l"],
                "metas_batch": ["m"],
                "vocal_languages_batch": ["en"],
                "target_wavs_tensor": None,
                "repainting_start_batch": [0.0],
                "repainting_end_batch": [1.0],
                "instructions_batch": ["i"],
                "audio_code_hints_batch": None,
                "should_return_intermediate": True,
            },
            refer_audios=None,
            guidance_scale=7.0,
            actual_seed_list=[1],
            audio_cover_strength=1.0,
            cover_noise_strength=0.0,
            guidance_mode="apg",
            cfg_interval_start=0.0,
            cfg_interval_end=1.0,
            shift=1.0,
            infer_method="ode",
        )
        self.assertEqual(host.service_calls, 1)
        self.assertEqual(out["outputs"]["target_latents"], "ok")

    def test_progress_is_passed_to_service_generate(self):
        """The progress callback should be forwarded to service_generate."""
        host = _Host()
        progress_fn = lambda *args, **kwargs: None
        host._run_generate_music_service_with_progress(
            progress=progress_fn,
            actual_batch_size=1,
            audio_duration=10.0,
            inference_steps=8,
            timesteps=None,
            service_inputs={
                "captions_batch": ["c"],
                "lyrics_batch": ["l"],
                "metas_batch": ["m"],
                "vocal_languages_batch": ["en"],
                "target_wavs_tensor": None,
                "repainting_start_batch": [0.0],
                "repainting_end_batch": [1.0],
                "instructions_batch": ["i"],
                "audio_code_hints_batch": None,
                "should_return_intermediate": True,
            },
            refer_audios=None,
            guidance_scale=7.0,
            actual_seed_list=[1],
            audio_cover_strength=1.0,
            cover_noise_strength=0.0,
            guidance_mode="apg",
            cfg_interval_start=0.0,
            cfg_interval_end=1.0,
            shift=1.0,
            infer_method="ode",
        )
        # Verify that progress was forwarded as a kwarg to service_generate
        self.assertIs(host.service_kwargs.get("progress"), progress_fn)


if __name__ == "__main__":
    unittest.main()
