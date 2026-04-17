"""HTTP routes for SuperSep multi-stage stem separation."""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Callable, Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger


def register_supersep_routes(
    app: FastAPI,
    *,
    get_project_root: Callable[[], str],
) -> None:
    """Register SuperSep stem separation endpoints."""

    _jobs: Dict[str, Dict] = {}
    _lock = Lock()
    _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="supersep")

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

    @app.post("/v1/supersep/separate")
    async def supersep_separate(request: Request):
        """Start a SuperSep separation job.

        Body: {audio_path, level: "basic"|"vocal-split"|"full"|"maximum"}
        Returns: {job_id, status: "running"}
        """
        body = await request.json()
        audio_path = body.get("audio_path") or body.get("audioPath", "")
        level = body.get("level", "full")

        if not audio_path:
            raise HTTPException(400, "audio_path is required")

        audio_path = _resolve_audio_path(audio_path)

        if not os.path.isfile(audio_path):
            raise HTTPException(404, f"Audio file not found: {audio_path}")

        valid_levels = ["basic", "vocal-split", "full", "maximum"]
        if level not in valid_levels:
            raise HTTPException(400, f"level must be one of: {valid_levels}")

        job_id = str(uuid4())
        with _lock:
            _jobs[job_id] = {
                "status": "running",
                "stems": [],
                "error": None,
                "progress": 0.0,
                "stage": 0,
                "message": "Starting SuperSep pipeline...",
                "level": level,
            }

        def _run():
            # ── VRAM offloading: move ACE-Step models to CPU ──
            handler = getattr(app.state, "handler", None)
            offloaded_parts = []
            if handler and getattr(handler, "_models_loaded", False):
                import torch
                try:
                    for attr_name in ("model", "vae", "tokenizer"):
                        mod = getattr(handler, attr_name, None)
                        if mod is not None and hasattr(mod, "to"):
                            dev = next(mod.parameters()).device if hasattr(mod, "parameters") else None
                            if dev is not None and dev.type != "cpu":
                                logger.info(f"[SuperSep] Offloading {attr_name} from {dev} → cpu")
                                mod.to("cpu")
                                offloaded_parts.append((attr_name, mod, dev))
                    torch.cuda.empty_cache()
                    logger.info("[SuperSep] ACE-Step models offloaded to CPU")
                except Exception as e:
                    logger.warning(f"[SuperSep] VRAM offload failed (non-fatal): {e}")

            try:
                from acestep.supersep_pipeline import run_supersep

                project_root = get_project_root()
                output_dir = os.path.join(
                    project_root, ".cache", "acestep", "supersep", job_id,
                )

                def _cb(stage: int, msg: str, pct: float):
                    with _lock:
                        if job_id in _jobs:
                            _jobs[job_id]["stage"] = stage
                            _jobs[job_id]["message"] = msg
                            _jobs[job_id]["progress"] = pct

                stems = run_supersep(
                    audio_path,
                    output_dir,
                    level=level,
                    progress_callback=_cb,
                )

                with _lock:
                    _jobs[job_id]["status"] = "complete"
                    _jobs[job_id]["progress"] = 1.0
                    _jobs[job_id]["message"] = "Done"
                    _jobs[job_id]["stems"] = [s.to_dict() for s in stems]

            except Exception as e:
                logger.error(f"[SuperSep] Job {job_id} failed: {e}")
                with _lock:
                    _jobs[job_id]["status"] = "failed"
                    _jobs[job_id]["error"] = str(e)
                    _jobs[job_id]["message"] = f"Error: {e}"
            finally:
                # ── Restore ACE-Step models to GPU ──
                if offloaded_parts:
                    import torch
                    for attr_name, mod, orig_dev in offloaded_parts:
                        try:
                            logger.info(f"[SuperSep] Restoring {attr_name} → {orig_dev}")
                            mod.to(orig_dev)
                        except Exception as e:
                            logger.warning(f"[SuperSep] Failed to restore {attr_name}: {e}")
                    torch.cuda.empty_cache()
                    logger.info("[SuperSep] ACE-Step models restored to GPU")

        _executor.submit(_run)
        return {"job_id": job_id, "status": "running"}

    @app.get("/v1/supersep/{job_id}/progress")
    async def supersep_progress(job_id: str):
        """SSE stream of SuperSep separation progress."""

        async def _event_stream():
            while True:
                with _lock:
                    job = _jobs.get(job_id)
                if job is None:
                    yield f'data: {{"type": "error", "message": "Job not found"}}\n\n'
                    return

                status = job["status"]
                if status == "complete":
                    yield f'data: {{"type": "complete", "stems": {json.dumps(job["stems"])}}}\n\n'
                    return
                elif status == "failed":
                    err = job.get("error", "Unknown error").replace('"', '\\"')
                    yield f'data: {{"type": "error", "message": "{err}"}}\n\n'
                    return
                else:
                    msg = job.get("message", "").replace('"', '\\"')
                    yield f'data: {{"type": "progress", "stage": {job["stage"]}, "percent": {job["progress"]:.3f}, "message": "{msg}"}}\n\n'

                await asyncio.sleep(0.5)

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/supersep/recombine")
    async def supersep_recombine(request: Request):
        """Recombine stems with volume/mute settings.

        Body: {stems: [{path, volume, muted}]}
        Returns: {mixed_path: "<absolute path to mixed file>"}
        """
        body = await request.json()
        stems = body.get("stems", [])

        if not stems:
            raise HTTPException(400, "stems array is required")

        # Validate all paths exist
        for s in stems:
            if not s.get("muted", False) and not os.path.isfile(s.get("path", "")):
                raise HTTPException(404, f"Stem file not found: {s.get('path', '')}")

        try:
            from acestep.supersep_pipeline import recombine_stems

            project_root = get_project_root()
            mix_id = str(uuid4())
            output_path = os.path.join(
                project_root, ".cache", "acestep", "supersep", "mixes", f"{mix_id}.flac",
            )

            mixed_path = recombine_stems(stems, output_path)
            return {"mixed_path": mixed_path}

        except Exception as e:
            logger.error(f"[SuperSep] Recombine failed: {e}")
            raise HTTPException(500, f"Recombination failed: {e}")

    @app.get("/v1/supersep/serve")
    async def supersep_serve(path: str = ""):
        """Serve a stem audio file from the supersep cache.

        Only allows serving files from .cache/acestep/supersep/ for security.
        """
        if not path:
            raise HTTPException(400, "path is required")

        # Security: resolve and verify the file is inside the supersep cache
        project_root = get_project_root()
        allowed_base = os.path.normpath(
            os.path.join(project_root, ".cache", "acestep", "supersep")
        )
        resolved = os.path.normpath(os.path.abspath(path))

        if not resolved.startswith(allowed_base):
            raise HTTPException(403, "Access denied: path outside supersep cache")

        if not os.path.isfile(resolved):
            raise HTTPException(404, f"File not found: {path}")

        # Determine media type from extension
        ext = os.path.splitext(resolved)[1].lower()
        media_types = {
            ".flac": "audio/flac",
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg",
        }
        media_type = media_types.get(ext, "application/octet-stream")

        return FileResponse(resolved, media_type=media_type)
