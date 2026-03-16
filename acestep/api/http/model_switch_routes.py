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
        current_lm = os.getenv("ACESTEP_LM_MODEL_PATH", "").strip() or None
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
