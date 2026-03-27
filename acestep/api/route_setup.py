"""Application route and middleware registration helpers for API server."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import UploadFile as StarletteUploadFile

from acestep.api.http.audio_route import register_audio_route
from acestep.api.http.cancel_route import register_cancel_route
from acestep.api.http.enhance_routes import register_enhance_routes
from acestep.api.http.lora_routes import register_lora_routes
from acestep.api.http.lm_lora_routes import register_lm_lora_routes
from acestep.api.http.model_service_routes import register_model_service_routes
from acestep.api.http.model_switch_routes import register_model_switch_routes
from acestep.api.http.query_result_route import register_query_result_route
from acestep.api.http.reinitialize_route import register_reinitialize_route
from acestep.api.http.release_task_route import register_release_task_route
from acestep.api.http.sample_format_routes import register_sample_format_routes
from acestep.api.http.stats_route import register_stats_route
from acestep.api.http.steering_routes import register_steering_routes
from acestep.api.http.stems_routes import register_stems_routes
from acestep.api.http.system_routes import register_system_routes
from acestep.api.http.mastering_routes import register_mastering_routes
from acestep.api.http.cover_art_routes import register_cover_art_routes
from acestep.api.http.llm_routes import register_llm_routes
from acestep.api.http.lireek_routes import register_lireek_routes
from acestep.openrouter_adapter import create_openrouter_router


def configure_api_routes(
    app: FastAPI,
    *,
    store: Any,
    queue_maxsize: int,
    initial_avg_job_seconds: float,
    verify_api_key: Callable[..., Any],
    verify_token_from_request: Callable[[dict, Optional[str]], Optional[str]],
    wrap_response: Callable[..., Dict[str, Any]],
    get_project_root: Callable[[], str],
    get_model_name: Callable[[str], str],
    ensure_model_downloaded: Callable[[str, str], str],
    env_bool: Callable[[str, bool], bool],
    simple_example_data: List[Dict[str, Any]],
    custom_example_data: List[Dict[str, Any]],
    format_sample: Callable[..., Any],
    to_int: Callable[[Any, Optional[int]], Optional[int]],
    to_float: Callable[[Any, Optional[float]], Optional[float]],
    request_parser_cls: Any,
    request_model_cls: Any,
    validate_audio_path: Callable[[Optional[str]], Optional[str]],
    save_upload_to_temp: Callable[..., Any],
    default_dit_instruction: str,
    lm_default_temperature: float,
    lm_default_cfg_scale: float,
    lm_default_top_p: float,
    map_status: Callable[[str], int],
    result_key_prefix: str,
    task_timeout_seconds: int,
    log_buffer: Any,
) -> None:
    """Configure middleware, compatibility router, and all API route modules."""

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["null", "http://localhost", "http://127.0.0.1"],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    openrouter_router = create_openrouter_router(lambda: app.state)
    app.include_router(openrouter_router)

    # ── Upstream route modules ─────────────────────────────────────

    register_model_service_routes(
        app=app,
        verify_api_key=verify_api_key,
        wrap_response=wrap_response,
        store=store,
        queue_maxsize=queue_maxsize,
        initial_avg_job_seconds=initial_avg_job_seconds,
        get_project_root=get_project_root,
        get_model_name=get_model_name,
        ensure_model_downloaded=ensure_model_downloaded,
        env_bool=env_bool,
    )

    register_sample_format_routes(
        app=app,
        verify_token_from_request=verify_token_from_request,
        wrap_response=wrap_response,
        simple_example_data=simple_example_data,
        custom_example_data=custom_example_data,
        format_sample=format_sample,
        get_project_root=get_project_root,
        get_model_name=get_model_name,
        ensure_model_downloaded=ensure_model_downloaded,
        env_bool=env_bool,
        to_int=to_int,
        to_float=to_float,
    )

    register_lora_routes(
        app=app,
        verify_api_key=verify_api_key,
        wrap_response=wrap_response,
        get_project_root=get_project_root,
    )

    register_lm_lora_routes(
        app=app,
        verify_api_key=verify_api_key,
        wrap_response=wrap_response,
    )

    register_reinitialize_route(
        app=app,
        verify_api_key=verify_api_key,
        wrap_response=wrap_response,
        env_bool=env_bool,
        get_project_root=get_project_root,
    )


    register_audio_route(app=app, verify_api_key=verify_api_key)

    register_release_task_route(
        app=app,
        verify_token_from_request=verify_token_from_request,
        wrap_response=wrap_response,
        store=store,
        request_parser_cls=request_parser_cls,
        request_model_cls=request_model_cls,
        validate_audio_path=validate_audio_path,
        save_upload_to_temp=save_upload_to_temp,
        upload_file_type=StarletteUploadFile,
        default_dit_instruction=default_dit_instruction,
        lm_default_temperature=lm_default_temperature,
        lm_default_cfg_scale=lm_default_cfg_scale,
        lm_default_top_p=lm_default_top_p,
    )

    register_query_result_route(
        app=app,
        verify_token_from_request=verify_token_from_request,
        wrap_response=wrap_response,
        store=store,
        map_status=map_status,
        result_key_prefix=result_key_prefix,
        task_timeout_seconds=task_timeout_seconds,
        log_buffer=log_buffer,
    )

    # ── Custom route modules (our additions) ───────────────────────

    register_cancel_route(
        app=app,
        verify_token_from_request=verify_token_from_request,
        wrap_response=wrap_response,
        store=store,
    )

    register_stats_route(
        app=app,
        verify_api_key=verify_api_key,
        wrap_response=wrap_response,
        store=store,
        queue_maxsize=queue_maxsize,
        initial_avg_job_seconds=initial_avg_job_seconds,
    )

    register_model_switch_routes(
        app=app,
        verify_api_key=verify_api_key,
        wrap_response=wrap_response,
        get_model_name=get_model_name,
    )

    register_steering_routes(
        app=app,
        verify_api_key=verify_api_key,
        verify_token_from_request=verify_token_from_request,
        wrap_response=wrap_response,
    )

    register_system_routes(
        app=app,
        log_buffer=log_buffer,
    )

    register_stems_routes(
        app=app,
        get_project_root=get_project_root,
    )

    register_enhance_routes(
        app=app,
        get_project_root=get_project_root,
    )

    register_mastering_routes(
        app=app,
        get_project_root=get_project_root,
    )

    register_cover_art_routes(
        app=app,
        get_project_root=get_project_root,
    )

    # ── Lireek integration routes ──────────────────────────────────

    # Initialise Lireek database on first startup
    from acestep.api.lireek.lireek_db import init_db as init_lireek_db
    init_lireek_db()

    register_llm_routes(app=app)
    register_lireek_routes(app=app)

