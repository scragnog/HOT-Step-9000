"""HTTP routes for model hot-swap and listing."""

from __future__ import annotations

from typing import Any, Callable, Dict

from fastapi import Depends, FastAPI, HTTPException, Request

from acestep.handler import AceStepHandler


def register_model_switch_routes(
    app: FastAPI,
    *,
    verify_api_key: Callable[..., Any],
    wrap_response: Callable[..., Dict[str, Any]],
    get_model_name: Callable[[str], str],
) -> None:
    """Register model listing and hot-swap endpoints."""

    @app.get("/v1/models/list")
    async def list_models(_: None = Depends(verify_api_key)):
        """List available DiT models (includes all downloadable models)."""
        current_model = get_model_name(app.state._config_path) if getattr(app.state, "_initialized", False) else None

        # Scan checkpoints directory for installed models
        installed = set()
        h: AceStepHandler = app.state.handler
        if h:
            installed = set(h.get_available_acestep_v15_models())

        # Pre-loaded secondary handlers
        preloaded = set()
        if getattr(app.state, "_initialized2", False) and app.state._config_path2:
            preloaded.add(get_model_name(app.state._config_path2))
        if getattr(app.state, "_initialized3", False) and app.state._config_path3:
            preloaded.add(get_model_name(app.state._config_path3))

        models = []
        candidate_names = {name for name in installed if name}
        if current_model:
            candidate_names.add(current_model)
        candidate_names.update(name for name in preloaded if name)

        for name in sorted(candidate_names):
            is_active = name == current_model
            models.append({
                "name": name,
                "is_active": is_active,
                "is_preloaded": name in preloaded or is_active,
                # Backward-compatible alias for older clients.
                "is_default": is_active,
            })

        return wrap_response({
            "models": models,
            "active_model": current_model,
        })

    @app.get("/v1/models/status")
    async def list_models_status():
        """Lightweight model status check (no auth required).

        Used by loading.html to detect when models are fully loaded.
        Returns active_model: null until initialization is complete.
        """
        import os
        current_model = get_model_name(app.state._config_path) if getattr(app.state, "_initialized", False) else None
        # Use actually loaded LM model (may differ from env after hot-switch)
        llm = getattr(app.state, "llm_handler", None)
        prev_params = getattr(llm, "last_init_params", None) if llm else None
        current_lm = (
            (prev_params or {}).get("lm_model_path", "")
            if isinstance(prev_params, dict) else ""
        ) or os.getenv("ACESTEP_LM_MODEL_PATH", "").strip() or None
        return wrap_response({
            "active_model": current_model,
            "lm_model": current_lm,
        })

    @app.post("/v1/models/switch")
    async def switch_model_endpoint(request: Request, _: None = Depends(verify_api_key)):
        """Explicitly switch the primary handler's DiT model."""
        handler: AceStepHandler = app.state.handler
        if handler is None:
            raise HTTPException(status_code=500, detail="Handler not initialized")

        body = await request.json()
        target_model = body.get("model")
        if not target_model:
            raise HTTPException(status_code=400, detail="'model' field is required")

        current_model = get_model_name(app.state._config_path) if getattr(app.state, "_initialized", False) else None
        if target_model == current_model:
            return wrap_response({
                "message": f"Model '{target_model}' is already active",
                "active_model": current_model,
                "switched": False,
            })

        use_flash = getattr(app.state, "_use_flash_attention", True)
        status_msg, ok = handler.switch_dit_model(target_model, use_flash_attention=use_flash)
        if ok:
            app.state._config_path = target_model
            return wrap_response({
                "message": status_msg,
                "active_model": target_model,
                "switched": True,
            })
        else:
            raise HTTPException(status_code=500, detail=status_msg)

    @app.post("/v1/models/lm/switch")
    async def switch_lm_model_endpoint(request: Request, _: None = Depends(verify_api_key)):
        """Hot-switch the LM (language model) used for CoT / thinking."""
        import os

        body = await request.json()
        target_model = (body.get("model") or "").strip()
        if not target_model:
            raise HTTPException(status_code=400, detail="'model' field is required")

        llm = getattr(app.state, "llm_handler", None)
        if llm is None:
            raise HTTPException(status_code=500, detail="LLM handler not available")

        # Check current model
        prev_params = getattr(llm, "last_init_params", None)
        current_lm = (
            (prev_params or {}).get("lm_model_path", "")
            if isinstance(prev_params, dict) else ""
        )
        if current_lm and current_lm.strip() == target_model:
            return wrap_response({
                "message": f"LM model '{target_model}' is already active",
                "lm_model": target_model,
                "switched": False,
            })

        lock = getattr(app.state, "_llm_init_lock", None)
        if lock is None:
            raise HTTPException(status_code=500, detail="LLM init lock not available")

        from acestep.model_downloader import ensure_lm_model

        with lock:
            # Unload current model
            try:
                llm.unload()
            except Exception as exc:
                print(f"[API Server] Warning: LM unload failed: {exc}")
            app.state._llm_initialized = False
            app.state._llm_init_error = None

            # Resolve checkpoint dir and download if needed
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            checkpoint_dir = os.path.join(project_root, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

            lm_model_name = get_model_name(target_model)
            if lm_model_name:
                try:
                    ensure_lm_model(lm_model_name, checkpoint_dir)
                except Exception as exc:
                    print(f"[API Server] Warning: LM model download failed: {exc}")

            # Build init params from previous or defaults
            new_params = dict(prev_params) if isinstance(prev_params, dict) else {
                "checkpoint_dir": checkpoint_dir,
                "backend": os.getenv("ACESTEP_LM_BACKEND", "vllm").strip().lower() or "vllm",
                "device": os.getenv("ACESTEP_LM_DEVICE", os.getenv("ACESTEP_DEVICE", "auto")),
                "offload_to_cpu": os.getenv("ACESTEP_LM_OFFLOAD_TO_CPU", "").lower() in {"1", "true", "yes"},
                "dtype": None,
            }
            new_params["checkpoint_dir"] = checkpoint_dir
            new_params["lm_model_path"] = target_model

            status_msg, ok = llm.initialize(**new_params)
            if ok:
                app.state._llm_initialized = True
                app.state._llm_init_error = None
                # Also update env so status endpoint stays consistent
                os.environ["ACESTEP_LM_MODEL_PATH"] = target_model
                return wrap_response({
                    "message": f"LM model switched to {target_model}",
                    "lm_model": target_model,
                    "switched": True,
                })
            else:
                app.state._llm_init_error = status_msg
                raise HTTPException(
                    status_code=500,
                    detail=f"LM model switch failed: {status_msg}",
                )
