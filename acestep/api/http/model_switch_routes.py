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
        no_init = getattr(app.state, "_no_init_mode", False)
        current_model = get_model_name(app.state._config_path) if getattr(app.state, "_initialized", False) or no_init else None

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
        no_init_mode = getattr(app.state, "_no_init_mode", False)
        current_model = get_model_name(app.state._config_path) if getattr(app.state, "_initialized", False) or no_init_mode else None
        # Use actually loaded LM model (may differ from env after hot-switch)
        llm = getattr(app.state, "llm_handler", None)
        prev_params = getattr(llm, "last_init_params", None) if llm else None
        current_lm = (
            (prev_params or {}).get("lm_model_path", "")
            if isinstance(prev_params, dict) else ""
        ) or os.getenv("ACESTEP_LM_MODEL_PATH", "").strip() or None

        no_init = os.getenv("ACESTEP_NO_INIT", "").strip().lower() in ("true", "1", "yes")

        # Current DiT quantization and compile state
        handler: AceStepHandler = app.state.handler
        dit_params = getattr(handler, "last_init_params", None) or {} if handler else {}
        current_quantization = dit_params.get("quantization", None)
        compile_model = dit_params.get("compile_model", False)

        return wrap_response({
            "active_model": current_model,
            "lm_model": current_lm,
            "no_init": no_init,
            "quantization": current_quantization,
            "compile_model": compile_model,
        })

    @app.post("/v1/models/switch")
    async def switch_model_endpoint(request: Request, _: None = Depends(verify_api_key)):
        """Explicitly switch the primary handler's DiT model and/or quantization."""
        handler: AceStepHandler = app.state.handler
        if handler is None:
            raise HTTPException(status_code=500, detail="Handler not initialized")

        body = await request.json()
        target_model = body.get("model")
        if not target_model:
            raise HTTPException(status_code=400, detail="'model' field is required")

        # Optional quantization change
        new_quantization = body.get("quantization", "_unchanged")
        quant_changing = new_quantization != "_unchanged"

        current_model = get_model_name(app.state._config_path) if getattr(app.state, "_initialized", False) else None
        if target_model == current_model and not quant_changing:
            return wrap_response({
                "message": f"Model '{target_model}' is already active",
                "active_model": current_model,
                "switched": False,
            })

        use_flash = getattr(app.state, "_use_flash_attention", True)

        # Build kwargs for switch_dit_model
        switch_kwargs: dict = {
            "config_path": target_model,
            "use_flash_attention": use_flash,
        }
        if quant_changing:
            # Map "none" string to None (= full precision)
            switch_kwargs["quantization"] = None if new_quantization in (None, "none") else new_quantization

        status_msg, ok = handler.switch_dit_model(**switch_kwargs)
        if ok:
            app.state._config_path = target_model
            dit_params = getattr(handler, "last_init_params", None) or {}
            return wrap_response({
                "message": status_msg,
                "active_model": target_model,
                "switched": True,
                "quantization": dit_params.get("quantization", None),
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
        prev_params = getattr(llm, "last_init_params", None) or {}
        if not isinstance(prev_params, dict):
            prev_params = {}
        current_lm = prev_params.get("lm_model_path", "").strip()
        if current_lm == target_model:
            return wrap_response({
                "message": f"LM model '{target_model}' is already active",
                "lm_model": target_model,
                "switched": False,
            })

        lock = getattr(app.state, "_llm_init_lock", None)
        if lock is None:
            raise HTTPException(status_code=500, detail="LLM init lock not available")

        with lock:
            # Unload current model
            try:
                llm.unload()
            except Exception as exc:
                print(f"[API Server] Warning: LM unload failed: {exc}")
            app.state._llm_initialized = False
            app.state._llm_init_error = None

            # Resolve checkpoint_dir from previous params or project root
            checkpoint_dir = prev_params.get("checkpoint_dir", "")
            if not checkpoint_dir:
                from acestep.model_downloader import get_checkpoints_dir
                checkpoint_dir = str(get_checkpoints_dir())

            # Ensure model is downloaded (skip if already exists)
            from acestep.model_downloader import check_model_exists, ensure_lm_model
            if not check_model_exists(target_model, checkpoint_dir):
                print(f"[LM Switch] Model '{target_model}' not found locally, downloading...")
                try:
                    ensure_lm_model(target_model, checkpoint_dir)
                except Exception as exc:
                    app.state._llm_init_error = str(exc)
                    raise HTTPException(
                        status_code=500,
                        detail=f"LM model download failed: {exc}",
                    )
            else:
                print(f"[LM Switch] Model '{target_model}' found locally, loading...")

            # Build init params — only the 6 kwargs that initialize() accepts
            init_kwargs = {
                "checkpoint_dir": checkpoint_dir,
                "lm_model_path": target_model,
                "backend": prev_params.get("backend", os.getenv("ACESTEP_LM_BACKEND", "vllm").strip().lower() or "vllm"),
                "device": prev_params.get("device", os.getenv("ACESTEP_LM_DEVICE", os.getenv("ACESTEP_DEVICE", "auto"))),
                "offload_to_cpu": prev_params.get("offload_to_cpu", False),
                "dtype": None,  # Let initialize() auto-detect
            }

            status_msg, ok = llm.initialize(**init_kwargs)
            if ok:
                app.state._llm_initialized = True
                app.state._llm_init_error = None
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

    @app.post("/v1/models/lm/backend")
    async def switch_lm_backend_endpoint(request: Request, _: None = Depends(verify_api_key)):
        """Hot-switch the LM backend (pt ↔ vllm) while keeping the same model."""
        import os

        body = await request.json()
        target_backend = (body.get("backend") or "").strip().lower()
        if target_backend not in ("pt", "vllm", "custom-vllm"):
            raise HTTPException(status_code=400, detail="'backend' must be 'pt', 'vllm', or 'custom-vllm'")

        llm = getattr(app.state, "llm_handler", None)
        if llm is None:
            raise HTTPException(status_code=500, detail="LLM handler not available")

        # Wait for any active generation to finish before touching the LLM.
        # Switching backends while CUDA is mid-inference corrupts the stream.
        generation_idle = getattr(app.state, "generation_idle", None)
        if generation_idle is not None:
            print("[API Server] Backend switch: waiting for active generation to finish...")
            if not generation_idle.wait(timeout=120):
                raise HTTPException(
                    status_code=409,
                    detail="Cannot switch backend: a generation is still running after 120s timeout",
                )
            print("[API Server] Backend switch: generation idle, proceeding")

        # Check if already using this backend
        prev_params = getattr(llm, "last_init_params", None) or {}
        if not isinstance(prev_params, dict):
            prev_params = {}
        current_backend = prev_params.get("backend", os.getenv("ACESTEP_LM_BACKEND", "vllm")).strip().lower()
        if current_backend == target_backend:
            return wrap_response({
                "message": f"LM backend is already '{target_backend}'",
                "backend": target_backend,
                "switched": False,
            })

        lock = getattr(app.state, "_llm_init_lock", None)
        if lock is None:
            raise HTTPException(status_code=500, detail="LLM init lock not available")

        with lock:
            # Unload current model
            try:
                llm.unload()
            except Exception as exc:
                print(f"[API Server] Warning: LM unload failed: {exc}")
            app.state._llm_initialized = False
            app.state._llm_init_error = None

            # Build init params — same model, new backend
            checkpoint_dir = prev_params.get("checkpoint_dir", "")
            if not checkpoint_dir:
                from acestep.model_downloader import get_checkpoints_dir
                checkpoint_dir = str(get_checkpoints_dir())

            lm_model = prev_params.get("lm_model_path", "") or os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B").strip()

            init_kwargs = {
                "checkpoint_dir": checkpoint_dir,
                "lm_model_path": lm_model,
                "backend": target_backend,
                "device": prev_params.get("device", os.getenv("ACESTEP_LM_DEVICE", os.getenv("ACESTEP_DEVICE", "auto"))),
                "offload_to_cpu": prev_params.get("offload_to_cpu", False),
                "dtype": None,
            }

            print(f"[API Server] Switching LM backend: {current_backend} → {target_backend}")
            status_msg, ok = llm.initialize(**init_kwargs)
            if ok:
                app.state._llm_initialized = True
                app.state._llm_init_error = None
                os.environ["ACESTEP_LM_BACKEND"] = target_backend
                return wrap_response({
                    "message": f"LM backend switched to {target_backend}",
                    "backend": target_backend,
                    "switched": True,
                })
            else:
                app.state._llm_init_error = status_msg
                raise HTTPException(
                    status_code=500,
                    detail=f"LM backend switch failed: {status_msg}",
                )
