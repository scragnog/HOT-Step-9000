"""HTTP route for cancelling generation jobs."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request


def register_cancel_route(
    app: FastAPI,
    *,
    verify_token_from_request: Callable[[dict, Optional[str]], Optional[str]],
    wrap_response: Callable[..., Dict[str, Any]],
    store: Any,
) -> None:
    """Register the /cancel_task endpoint."""

    @app.post("/cancel_task")
    async def cancel_task_endpoint(request: Request, authorization: Optional[str] = Header(None)):
        """Cancel a queued or running generation job.

        For queued jobs: marks as cancelled so the worker skips it when dequeued.
        For running jobs: marks as failed immediately (mid-diffusion interrupt is not supported;
        the running inference step will complete but the result will be discarded).
        """
        content_type = (request.headers.get("content-type") or "").lower()
        if "json" in content_type:
            body = await request.json()
        else:
            form = await request.form()
            body = {k: v for k, v in form.items()}

        verify_token_from_request(body, authorization)
        job_id = body.get("job_id") or body.get("task_id")
        if not job_id:
            raise HTTPException(status_code=400, detail="'job_id' is required")

        # Flag it so the worker skips it if still queued
        app.state.cancelled_jobs.add(job_id)

        # Also remove from pending_ids so queue position reporting is correct
        async with app.state.pending_lock:
            try:
                app.state.pending_ids.remove(job_id)
            except ValueError:
                pass

        # Mark the job store record immediately (covers running jobs too)
        rec = store.get(job_id)
        if rec and rec.status not in ("succeeded", "failed"):
            store.mark_failed(job_id, "Cancelled by user")
            # Signal the LLM handler to stop any in-progress token generation
            llm_handler = getattr(app.state, "llm_handler", None)
            if llm_handler is not None:
                llm_handler._cancel_requested = True
            # Update local cache if available
            local_cache = getattr(app.state, "local_cache", None)
            if local_cache is not None:
                import json
                import time

                local_cache.set(f"ace_step_v1.5_{job_id}", json.dumps(
                    [{"file": "", "wave": "", "status": 2, "create_time": int(time.time()), "error": "Cancelled by user"}]
                ))
            print(f"[API Server] Job {job_id} cancelled by user")

        return wrap_response({"job_id": job_id, "status": "cancelled"})
