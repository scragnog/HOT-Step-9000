"""HTTP routes for mastering preset management and re-mastering."""

from __future__ import annotations

import json
import os
import uuid
from typing import Callable, Dict

from fastapi import FastAPI, HTTPException, Request
from loguru import logger


def register_mastering_routes(
    app: FastAPI,
    *,
    get_project_root: Callable[[], str],
) -> None:
    """Register mastering preset and re-master endpoints."""

    def _resolve_audio_path(audio_path: str) -> str:
        """Resolve HTTP audio URL paths to actual disk paths."""
        project_root = get_project_root()
        if audio_path.startswith("/audio/"):
            return os.path.join(project_root, "ace-step-ui", "server", "public", audio_path.lstrip("/"))
        elif audio_path.startswith("/v1/audio"):
            import urllib.parse as _urlparse
            parsed = _urlparse.urlparse(audio_path)
            qs = _urlparse.parse_qs(parsed.query)
            if "path" in qs:
                return qs["path"][0]
        elif not audio_path.startswith("http") and not os.path.isabs(audio_path):
            return os.path.join(project_root, audio_path)
        return audio_path

    @app.get("/v1/mastering/presets")
    async def list_mastering_presets():
        """List all available mastering presets."""
        from acestep.core.audio.mastering import MasteringEngine
        return {"presets": MasteringEngine.list_presets()}

    @app.post("/v1/mastering/presets")
    async def save_mastering_preset(request: Request):
        """Save a custom mastering preset."""
        body = await request.json()
        name = body.get("name", "").strip()
        params = body.get("params")

        if not name:
            raise HTTPException(400, "Preset name is required")
        if not params or not isinstance(params, dict):
            raise HTTPException(400, "Preset params dict is required")

        from acestep.core.audio.mastering import MasteringEngine
        preset_id = MasteringEngine.save_preset(name, params)
        return {"id": preset_id, "name": name}

    @app.delete("/v1/mastering/presets/{preset_id}")
    async def delete_mastering_preset(preset_id: str):
        """Delete a custom mastering preset."""
        from acestep.core.audio.mastering import MasteringEngine
        deleted = MasteringEngine.delete_preset(preset_id)
        if not deleted:
            raise HTTPException(404, f"Preset '{preset_id}' not found or protected")
        return {"deleted": preset_id}

    @app.post("/v1/mastering/apply")
    async def apply_mastering(request: Request):
        """Re-master an audio file with given parameters.

        Supports two modes:
          - 'builtin' (default): Parametric mastering via MasteringEngine
          - 'matchering': Reference-based mastering via the matchering library
            optionally with stem-by-stem matching (stem_matchering=True)

        Request body:
            audio_path: Path/URL to the original (unmastered) audio file
            mastering_params: Dict of mastering parameters to apply
        """
        body = await request.json()
        audio_path = body.get("audio_path", "")
        mastering_params = body.get("mastering_params") or {}

        if not audio_path:
            raise HTTPException(400, "audio_path is required")

        audio_path = _resolve_audio_path(audio_path)
        if not os.path.isfile(audio_path):
            raise HTTPException(404, f"Audio file not found: {audio_path}")

        requested_mode = mastering_params.get("mode", "builtin")

        try:
            if requested_mode == "matchering":
                return await _apply_matchering(audio_path, mastering_params, get_project_root)
            else:
                return _apply_builtin(audio_path, mastering_params)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[Mastering] Re-master failed: {e}", exc_info=True)
            raise HTTPException(500, f"Re-mastering failed: {e}")


def _apply_builtin(audio_path: str, mastering_params: dict) -> dict:
    """Parametric mastering via MasteringEngine."""
    import numpy as np
    import soundfile as sf
    from acestep.core.audio.mastering import MasteringEngine

    audio_data, sample_rate = sf.read(audio_path, dtype="float32")
    if audio_data.ndim == 1:
        audio_data = np.stack([audio_data, audio_data])
    else:
        audio_data = audio_data.T

    engine = MasteringEngine()
    mastered = engine.master(audio_data, sample_rate, params_override=mastering_params)

    base, ext = os.path.splitext(audio_path)
    uid_suffix = uuid.uuid4().hex[:8]
    if base.endswith("_original"):
        output_path = base.replace("_original", f"_remastered_{uid_suffix}") + ext
    else:
        output_path = base + f"_remastered_{uid_suffix}" + ext

    sf.write(output_path, mastered.T, sample_rate)
    logger.info(f"[Mastering] Re-mastered (builtin): {output_path}")
    return {"output_path": output_path, "sample_rate": sample_rate}


async def _apply_matchering(audio_path: str, mastering_params: dict, get_project_root) -> dict:
    """Reference-based mastering via the matchering library."""
    import asyncio
    import numpy as np
    import soundfile as sf
    import tempfile

    ref_path = mastering_params.get("reference_file") or mastering_params.get("reference")
    if not ref_path or not os.path.isfile(ref_path):
        raise HTTPException(400, f"Matchering reference file not found: {ref_path}")

    use_stem = mastering_params.get("stem_matchering", False)

    # Run the CPU-heavy matchering in a thread so we don't block the event loop
    def _do_matchering():
        import matchering as mg
        from acestep.audio_utils import ensure_mp3_for_matchering

        _, sample_rate = sf.read(audio_path, dtype="float32", frames=1)

        if use_stem:
            from acestep.inference import _run_stem_matchering
            logger.info("[Mastering] Running STEM Matchering pipeline (remaster)")

            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert non-MP3 references (FLAC, WAV, etc.) for Matchering
                effective_ref = ensure_mp3_for_matchering(ref_path, temp_dir)
                mastered = _run_stem_matchering(
                    audio_path, effective_ref, temp_dir, sample_rate, progress=None
                )
        else:
            logger.info("[Mastering] Running standard Matchering pipeline (remaster)")
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert non-MP3 references (FLAC, WAV, etc.) for Matchering
                effective_ref = ensure_mp3_for_matchering(ref_path, temp_dir)
                temp_out = os.path.join(temp_dir, "temp_out.wav")
                mg.process(
                    target=audio_path,
                    reference=effective_ref,
                    results=[mg.pcm16(temp_out)],
                )
                mastered, out_sr = sf.read(temp_out, dtype="float32")
                if mastered.ndim == 1:
                    mastered = np.column_stack((mastered, mastered))
                if out_sr != sample_rate:
                    import librosa
                    mastered = librosa.resample(
                        mastered.T, orig_sr=out_sr, target_sr=sample_rate
                    ).T

        # Save output
        base, ext = os.path.splitext(audio_path)
        uid_suffix = uuid.uuid4().hex[:8]
        suffix_tag = "stem_matchered" if use_stem else "matchered"
        if base.endswith("_original"):
            output_path = base.replace("_original", f"_{suffix_tag}_{uid_suffix}") + ext
        else:
            output_path = base + f"_{suffix_tag}_{uid_suffix}" + ext

        sf.write(output_path, mastered if mastered.ndim == 2 else mastered.T, sample_rate)
        logger.info(f"[Mastering] Matchered (stem={use_stem}): {output_path}")
        return {"output_path": output_path, "sample_rate": sample_rate}

    return await asyncio.get_event_loop().run_in_executor(None, _do_matchering)

