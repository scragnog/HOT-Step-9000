"""HTTP routes for stem separation."""

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


def register_stems_routes(
    app: FastAPI,
    *,
    get_project_root: Callable[[], str],
) -> None:
    """Register stem separation endpoints."""

    # Lazy-init stem service + in-memory job store
    _stem_service = None
    _stem_jobs: Dict[str, Dict] = {}
    _stem_lock = Lock()
    _stem_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stem")

    def _get_stem_service():
        nonlocal _stem_service
        if _stem_service is None:
            from acestep.stem_service import StemService
            _stem_service = StemService()
        return _stem_service

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

    @app.get("/v1/stems/available")
    async def stems_available():
        """Check if stem separation is available (audio_separator installed)."""
        try:
            svc = _get_stem_service()
            return {
                "available": svc.is_available(),
                "modes": ["vocals", "every-stem"],
            }
        except Exception:
            return {"available": False, "modes": []}

    @app.post("/v1/stems/separate")
    async def stems_separate(request: Request):
        """Start a stem separation job. Returns a job_id for polling progress."""
        body = await request.json()
        audio_path = body.get("audio_path") or body.get("audioPath", "")
        mode = body.get("mode", "two-pass")

        if not audio_path:
            raise HTTPException(400, "audio_path is required")

        audio_path = _resolve_audio_path(audio_path)

        if not os.path.isfile(audio_path):
            raise HTTPException(404, f"Audio file not found: {audio_path}")

        job_id = str(uuid4())
        with _stem_lock:
            _stem_jobs[job_id] = {
                "status": "running",
                "stems": [],
                "error": None,
                "progress": 0.0,
                "message": "Starting…",
            }

        def _run():
            # --- VRAM offloading: move ACE-Step models to CPU ---
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
                                logger.info(f"[Stem] Offloading {attr_name} from {dev} → cpu")
                                mod.to("cpu")
                                offloaded_parts.append((attr_name, mod, dev))
                    torch.cuda.empty_cache()
                    logger.info("[Stem] ACE-Step models offloaded to CPU for stem separation")
                except Exception as e:
                    logger.warning(f"[Stem] VRAM offload failed (non-fatal): {e}")

            try:
                svc = _get_stem_service()

                def _cb(msg: str, pct: float):
                    with _stem_lock:
                        if job_id in _stem_jobs:
                            _stem_jobs[job_id]["progress"] = pct
                            _stem_jobs[job_id]["message"] = msg

                stems = svc.separate(audio_path, mode=mode, progress_callback=_cb)
                with _stem_lock:
                    _stem_jobs[job_id]["status"] = "complete"
                    _stem_jobs[job_id]["progress"] = 1.0
                    _stem_jobs[job_id]["message"] = "Done"
                    _stem_jobs[job_id]["stems"] = [s.to_dict() for s in stems]
            except Exception as e:
                logger.error(f"[Stem] Job {job_id} failed: {e}")
                with _stem_lock:
                    _stem_jobs[job_id]["status"] = "failed"
                    _stem_jobs[job_id]["error"] = str(e)
                    _stem_jobs[job_id]["message"] = f"Error: {e}"
            finally:
                # --- Restore ACE-Step models to GPU ---
                if offloaded_parts:
                    import torch
                    for attr_name, mod, orig_dev in offloaded_parts:
                        try:
                            logger.info(f"[Stem] Restoring {attr_name} → {orig_dev}")
                            mod.to(orig_dev)
                        except Exception as e:
                            logger.warning(f"[Stem] Failed to restore {attr_name}: {e}")
                    torch.cuda.empty_cache()
                    logger.info("[Stem] ACE-Step models restored to GPU")

        _stem_executor.submit(_run)
        return {"job_id": job_id, "status": "running"}

    @app.get("/v1/stems/{job_id}/progress")
    async def stems_progress(job_id: str):
        """SSE stream of stem separation progress."""

        async def _event_stream():
            while True:
                with _stem_lock:
                    job = _stem_jobs.get(job_id)
                if job is None:
                    yield f'data: {{"type": "error", "message": "Job not found"}}\n\n'
                    return

                status = job["status"]
                if status == "complete":
                    yield f'data: {{"type": "complete", "stems": {json.dumps(job["stems"])}}}\n\n'
                    return
                elif status == "failed":
                    yield f'data: {{"type": "error", "message": "{job.get("error", "Unknown error")}"}}\n\n'
                    return
                else:
                    yield f'data: {{"type": "progress", "percent": {job["progress"]:.3f}, "message": "{job.get("message", "")}"}}\n\n'

                await asyncio.sleep(0.5)

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/v1/stems/{job_id}/download/{stem_name}")
    async def stems_download(job_id: str, stem_name: str):
        """Download a specific stem file from a completed job."""
        with _stem_lock:
            job = _stem_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        if job["status"] != "complete":
            raise HTTPException(400, f"Job not complete (status: {job['status']})")

        for stem in job["stems"]:
            if stem["stem_type"] == stem_name or stem["file_name"] == stem_name:
                fp = stem["file_path"]
                if os.path.isfile(fp):
                    return FileResponse(
                        fp,
                        media_type="audio/flac",
                        filename=stem["file_name"],
                    )
                raise HTTPException(404, f"Stem file missing from disk: {fp}")

        raise HTTPException(404, f"Stem '{stem_name}' not found in job")

    @app.get("/v1/stems/{job_id}/download_all")
    async def stems_download_all(job_id: str):
        """Download all stems for a completed job as a ZIP file."""
        import zipfile
        from starlette.background import BackgroundTask

        with _stem_lock:
            job = _stem_jobs.get(job_id)
            
        if job is None:
            raise HTTPException(404, "Job not found")
        if job["status"] != "complete":
            raise HTTPException(400, f"Job not complete (status: {job['status']})")

        stems = job["stems"]
        if not stems:
            raise HTTPException(400, "No stems generated for this job")

        # Use the directory of the first stem to store the temporary zip
        output_dir = os.path.dirname(stems[0]["file_path"])
        zip_path = os.path.join(output_dir, f"{job_id}_stems.zip")

        # Create zip file dynamically
        if not os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for stem in stems:
                    fp = stem["file_path"]
                    if os.path.isfile(fp):
                        zipf.write(fp, arcname=stem["file_name"])

        # Return the zipped file and schedule cleanup to preserve disk space
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="stems_bundle.zip",
            background=BackgroundTask(lambda: os.remove(zip_path) if os.path.exists(zip_path) else None)
        )

