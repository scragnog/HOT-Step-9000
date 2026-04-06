"""Startup model initialization orchestration for API server lifespan."""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from acestep.gpu_config import (
    VRAM_AUTO_OFFLOAD_THRESHOLD_GB,
    get_gpu_config,
    set_global_gpu_config,
)
from acestep.api.startup_llm_init import initialize_llm_at_startup


def initialize_models_at_startup(
    *,
    app: Any,
    handler: Any,
    llm_handler: Any,
    handler2: Any,
    handler3: Any,
    config_path2: str,
    config_path3: str,
    get_project_root: Callable[[], str],
    get_model_name: Callable[[str], str],
    ensure_model_downloaded: Callable[[str, str], str],
    env_bool: Callable[[str, bool], bool],
) -> None:
    """Initialize DiT and optional LLM models at server startup."""

    no_init = env_bool("ACESTEP_NO_INIT", False)
    gpu_config = get_gpu_config()
    set_global_gpu_config(gpu_config)
    app.state.gpu_config = gpu_config

    gpu_memory_gb = gpu_config.gpu_memory_gb
    auto_offload = gpu_memory_gb > 0 and gpu_memory_gb < VRAM_AUTO_OFFLOAD_THRESHOLD_GB

    print(f"\n{'='*60}")
    print("[API Server] GPU Configuration Detected:")
    print(f"{'='*60}")
    print(f"  GPU Memory: {gpu_memory_gb:.2f} GB")
    print(f"  Configuration Tier: {gpu_config.tier}")
    print(f"  Max Duration (with LM): {gpu_config.max_duration_with_lm}s")
    print(f"  Max Duration (without LM): {gpu_config.max_duration_without_lm}s")
    print(f"  Max Batch Size (with LM): {gpu_config.max_batch_size_with_lm}")
    print(f"  Max Batch Size (without LM): {gpu_config.max_batch_size_without_lm}")
    print(f"  Default LM Init: {gpu_config.init_lm_default}")
    print(f"  Available LM Models: {gpu_config.available_lm_models or 'None'}")
    print(f"{'='*60}\n")

    if no_init:
        print("[API Server] --no-init mode: Skipping all model loading at startup")
        print("[API Server] Models will be lazy-loaded on first request")
        print("[API Server] Server is ready to accept requests (models not loaded yet)")
        app.state._no_init_mode = True
        return

    print("[API Server] Initializing models at startup...")
    if auto_offload:
        print("[API Server] Auto-enabling CPU offload (GPU < 16GB)")
    elif gpu_memory_gb > 0:
        print("[API Server] CPU offload disabled by default (GPU >= 16GB)")
    else:
        print("[API Server] No GPU detected, running on CPU")

    project_root = get_project_root()
    config_path = os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
    device = os.getenv("ACESTEP_DEVICE", "auto")
    use_flash_attention = env_bool("ACESTEP_USE_FLASH_ATTENTION", True)

    offload_to_cpu_env = os.getenv("ACESTEP_OFFLOAD_TO_CPU")
    if offload_to_cpu_env is not None:
        offload_to_cpu = env_bool("ACESTEP_OFFLOAD_TO_CPU", False)
    else:
        offload_to_cpu = auto_offload
        if auto_offload:
            print("[API Server] Auto-setting offload_to_cpu=True based on GPU memory")

    offload_dit_to_cpu = env_bool("ACESTEP_OFFLOAD_DIT_TO_CPU", False)
    compile_model = env_bool("ACESTEP_COMPILE_MODEL", False)

    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    dit_model_name = get_model_name(config_path)
    if dit_model_name:
        try:
            ensure_model_downloaded(dit_model_name, checkpoint_dir)
        except Exception as exc:
            print(f"[API Server] Warning: Failed to download DiT model: {exc}")

    try:
        ensure_model_downloaded("vae", checkpoint_dir)
    except Exception as exc:
        print(f"[API Server] Warning: Failed to download VAE model: {exc}")

    # Read quantization preference from env: "auto", "int8_weight_only",
    # "int4_weight_only", or "none".  "auto" (or unset) uses the GPU tier
    # default; "none" explicitly disables quantization.
    quant_env = os.getenv("ACESTEP_QUANTIZATION", "auto").strip().lower()
    if quant_env in ("none", "off", "false", ""):
        quantization = None
    elif quant_env == "auto":
        # Mirror the Gradio pipeline: tier-based default
        quantization = "int8_weight_only" if gpu_config.quantization_default else None
    elif quant_env in ("int8_weight_only", "int4_weight_only"):
        quantization = quant_env
    else:
        print(f"[API Server] Warning: Unknown ACESTEP_QUANTIZATION='{quant_env}', defaulting to auto")
        quantization = "int8_weight_only" if gpu_config.quantization_default else None

    # torchao quantization requires torch.compile
    if quantization is not None:
        compile_model = True

    if quantization:
        print(f"[API Server] DiT quantization: {quantization}")

    print(f"[API Server] Loading primary DiT model: {config_path}")
    status_msg, ok = handler.initialize_service(
        project_root=project_root,
        config_path=config_path,
        device=device,
        use_flash_attention=use_flash_attention,
        compile_model=compile_model,
        offload_to_cpu=offload_to_cpu,
        offload_dit_to_cpu=offload_dit_to_cpu,
        quantization=quantization,
    )
    if not ok:
        app.state._init_error = status_msg
        print(f"[API Server] ERROR: Primary model failed to load: {status_msg}")
        print()
        print("=" * 60)
        print("  MODEL LOAD FAILED — How to fix:")
        print("=" * 60)
        print("  This usually means the model files are missing or corrupt.")
        print()
        print("  Try one of these solutions:")
        print("    1. Re-run install.bat (recommended — re-downloads models)")
        print("    2. Run: python -m acestep.model_downloader --force")
        print("    3. Delete the checkpoints/ folder and re-run install.bat")
        print("=" * 60)
        print()
        raise RuntimeError(status_msg)
    app.state._initialized = True
    print(f"[API Server] Primary model loaded: {get_model_name(config_path)}")

    # ── Redmond Mode: merge DPO quality adapter into decoder ────────
    redmond_enabled = env_bool("ACESTEP_REDMOND_MODE", False)
    redmond_scale = float(os.getenv("ACESTEP_REDMOND_SCALE", "0.7"))
    redmond_path = os.path.join(checkpoint_dir, "redmond-refine", "standard")

    if redmond_enabled and handler.quantization is None and os.path.isdir(redmond_path):
        print(f"[API Server] Redmond Mode: merging quality adapter at scale {redmond_scale:.2f}...")
        try:
            from acestep.core.generation.handler.redmond_mode import apply_redmond_at_startup
            result = apply_redmond_at_startup(handler, redmond_path, redmond_scale)
            print(f"[API Server] {result}")
        except Exception as exc:
            print(f"[API Server] Warning: Redmond Mode failed to apply: {exc}")
    elif redmond_enabled and handler.quantization is None and not os.path.isdir(redmond_path):
        # Auto-download the adapter
        print("[API Server] Redmond Mode: adapter not found, downloading...")
        print("[API Server]   Source: artificialguybr/AceStep_Refine_Redmond")
        try:
            from huggingface_hub import snapshot_download
            redmond_parent = os.path.join(checkpoint_dir, "redmond-refine")
            os.makedirs(redmond_parent, exist_ok=True)
            snapshot_download(
                "artificialguybr/AceStep_Refine_Redmond",
                allow_patterns="standard/*",
                local_dir=redmond_parent,
            )
            if os.path.isdir(redmond_path):
                print("[API Server] Redmond Mode: download complete, merging...")
                from acestep.core.generation.handler.redmond_mode import apply_redmond_at_startup
                result = apply_redmond_at_startup(handler, redmond_path, redmond_scale)
                print(f"[API Server] {result}")
            else:
                print("[API Server] Warning: Redmond adapter download succeeded but standard/ folder not found")
        except Exception as exc:
            print(f"[API Server] Warning: Redmond adapter download failed: {exc}")
            print("[API Server]   You can download manually via install.bat")
    else:
        # Store adapter path for potential runtime activation
        handler._redmond_adapter_path = redmond_path if os.path.isdir(redmond_path) else ""

    if handler2 and config_path2:
        model2_name = get_model_name(config_path2)
        if model2_name:
            try:
                ensure_model_downloaded(model2_name, checkpoint_dir)
            except Exception as exc:
                print(f"[API Server] Warning: Failed to download secondary model: {exc}")
        print(f"[API Server] Loading secondary DiT model: {config_path2}")
        try:
            status_msg2, ok2 = handler2.initialize_service(
                project_root=project_root,
                config_path=config_path2,
                device=device,
                use_flash_attention=use_flash_attention,
                compile_model=compile_model,
                offload_to_cpu=offload_to_cpu,
                offload_dit_to_cpu=offload_dit_to_cpu,
            )
            app.state._initialized2 = ok2
            if ok2:
                print(f"[API Server] Secondary model loaded: {model2_name}")
            else:
                print(f"[API Server] Warning: Secondary model failed: {status_msg2}")
        except Exception as exc:
            print(f"[API Server] Warning: Failed to initialize secondary model: {exc}")
            app.state._initialized2 = False

    if handler3 and config_path3:
        model3_name = get_model_name(config_path3)
        if model3_name:
            try:
                ensure_model_downloaded(model3_name, checkpoint_dir)
            except Exception as exc:
                print(f"[API Server] Warning: Failed to download third model: {exc}")
        print(f"[API Server] Loading third DiT model: {config_path3}")
        try:
            status_msg3, ok3 = handler3.initialize_service(
                project_root=project_root,
                config_path=config_path3,
                device=device,
                use_flash_attention=use_flash_attention,
                compile_model=compile_model,
                offload_to_cpu=offload_to_cpu,
                offload_dit_to_cpu=offload_dit_to_cpu,
            )
            app.state._initialized3 = ok3
            if ok3:
                print(f"[API Server] Third model loaded: {model3_name}")
            else:
                print(f"[API Server] Warning: Third model failed: {status_msg3}")
        except Exception as exc:
            print(f"[API Server] Warning: Failed to initialize third model: {exc}")
            app.state._initialized3 = False

    initialize_llm_at_startup(
        app=app,
        llm_handler=llm_handler,
        gpu_config=gpu_config,
        device=device,
        offload_to_cpu=offload_to_cpu,
        checkpoint_dir=checkpoint_dir,
        get_model_name=get_model_name,
        ensure_model_downloaded=ensure_model_downloaded,
        env_bool=env_bool,
    )

    print("[API Server] All models initialized successfully!")
