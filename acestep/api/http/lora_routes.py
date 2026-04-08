"""LoRA HTTP routes for adapter lifecycle controls.

Includes both basic PEFT-based endpoints and advanced multi-adapter
slot-based endpoints with group/layer scales and temporal scheduling.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field

from acestep.handler import AceStepHandler

_SUCCESS_PREFIX = "\u2705"
_WARNING_PREFIX = "\u26a0\ufe0f"


# ── Request models ──────────────────────────────────────────────────────

class LoadLoRARequest(BaseModel):
    lora_path: str = Field(..., description="Path to LoRA adapter directory or LoKr/LyCORIS safetensors file")
    adapter_name: Optional[str] = Field(default=None, description="Optional adapter name (uses path-derived name if omitted)")
    slot: Optional[int] = Field(default=None, description="Slot number for advanced multi-adapter mode (0-3)")
    scale: Optional[float] = Field(default=None, ge=0.0, le=4.0, description="Optional scale to apply immediately after load")
    group_scales: Optional[Dict[str, float]] = Field(default=None, description="Optional group scales {self_attn, cross_attn, mlp} to apply immediately after load")


class SetLoRAScaleRequest(BaseModel):
    adapter_name: Optional[str] = Field(default=None, description="Optional adapter name; defaults to active adapter")
    scale: float = Field(..., ge=0.0, le=4.0, description="LoRA scale (0.0-4.0)")
    slot: Optional[int] = Field(default=None, description="Slot number for advanced mode")


class ToggleLoRARequest(BaseModel):
    use_lora: bool = Field(..., description="Enable or disable LoRA")


class SetGroupScalesRequest(BaseModel):
    self_attn: float = Field(default=1.0, ge=0.0, le=4.0, description="Self-attention group scale")
    cross_attn: float = Field(default=1.0, ge=0.0, le=4.0, description="Cross-attention group scale")
    mlp: float = Field(default=1.0, ge=0.0, le=4.0, description="MLP/feed-forward group scale")
    cond_embed: float = Field(default=1.0, ge=0.0, le=4.0, description="Conditioning embedder group scale")


class SetSlotGroupScalesRequest(BaseModel):
    slot: int = Field(..., description="Slot number (0-3)")
    self_attn: float = Field(default=1.0, ge=0.0, le=4.0, description="Self-attention group scale")
    cross_attn: float = Field(default=1.0, ge=0.0, le=4.0, description="Cross-attention group scale")
    mlp: float = Field(default=1.0, ge=0.0, le=4.0, description="MLP/feed-forward group scale")
    cond_embed: float = Field(default=1.0, ge=0.0, le=4.0, description="Conditioning embedder group scale")


class SetSlotLayerScalesRequest(BaseModel):
    slot: int = Field(..., description="Slot number (0-3)")
    layer_scales: Dict[int, float] = Field(default_factory=dict, description="Layer index (0-23) -> scale (0.0-2.0). Omitted layers default to 1.0")


class SetSlotLayerScaleRequest(BaseModel):
    slot: int = Field(..., description="Slot number (0-3)")
    layer: int = Field(..., ge=0, le=23, description="Layer index (0-23)")
    scale: float = Field(..., ge=0.0, le=4.0, description="Scale value")


class SetTemporalScheduleRequest(BaseModel):
    """Set or clear a temporal adapter schedule for multi-singer switching."""
    clear: bool = Field(default=False, description="Set true to clear the schedule")
    slot_segments: Optional[Dict[int, List[Dict]]] = Field(
        default=None,
        description="Dict mapping slot ID to list of segments. Each segment: {start, end, scale, fade_in?, fade_out?}",
    )


class AudioDiffRequest(BaseModel):
    """Compute audio difference between reference and ablated tracks."""
    reference_path: str = Field(..., description="Path to the reference audio file")
    ablated_path: str = Field(..., description="Path to the ablated audio file")
    amplify: float = Field(default=3.0, ge=1.0, le=20.0, description="Amplification factor for the diff signal")


class SetSlotTriggerWordRequest(BaseModel):
    """Set trigger word and placement for an adapter slot (or basic mode)."""
    slot: Optional[int] = Field(default=None, description="Slot number for advanced mode; omit for basic single-adapter mode")
    trigger_word: str = Field(default="", description="Trigger word to inject into captions (empty string to clear)")
    tag_position: str = Field(default="prepend", description="Where to place the trigger word: prepend, append, or replace")


class RedmondToggleRequest(BaseModel):
    """Enable or disable Redmond Mode (DPO quality refinement)."""
    enabled: bool = Field(..., description="Whether to enable Redmond Mode")


class RedmondScaleRequest(BaseModel):
    """Set Redmond Mode scale."""
    scale: float = Field(..., ge=0.0, le=4.0, description="Redmond scale (0.0-4.0)")


# ── Helpers ─────────────────────────────────────────────────────────────

def _require_initialized_handler(app: FastAPI) -> AceStepHandler:
    """Return initialized handler or raise HTTP 500 when unavailable."""
    handler: AceStepHandler = app.state.handler
    if handler is None or handler.model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    return handler


def _is_success_message(result: str, allow_warning: bool = False) -> bool:
    """Check whether backend operation result is considered successful."""
    if result.startswith(_SUCCESS_PREFIX):
        return True
    return allow_warning and result.startswith(_WARNING_PREFIX)


# ── Route registration ─────────────────────────────────────────────────

def register_lora_routes(
    app: FastAPI,
    verify_api_key: Callable[..., Any],
    wrap_response: Callable[..., Dict[str, Any]],
    get_project_root: Callable[[], str] = lambda: "",
) -> None:
    """Register LoRA lifecycle endpoints on the provided FastAPI app."""

    # ── Basic endpoints ─────────────────────────────────────────────

    @app.post("/v1/lora/load")
    async def load_lora_endpoint(request: LoadLoRARequest, _: None = Depends(verify_api_key)):
        """Load LoRA adapter into the primary model."""
        handler = _require_initialized_handler(app)
        try:
            # Advanced mode: slot-based loading
            if request.slot is not None:
                result = handler.load_lora_slot(request.lora_path, slot=request.slot)
                if _is_success_message(result):
                    # Atomically apply scale + group_scales if provided
                    if request.scale is not None:
                        try:
                            handler.set_lora_slot_scale(request.scale, request.slot)
                        except Exception as exc:
                            logger.warning(f"Failed to apply initial scale for slot {request.slot}: {exc}")
                    if request.group_scales is not None:
                        logger.info(f"[LoRA Load] Applying group scales for slot {request.slot}: {request.group_scales}")
                        try:
                            handler.set_slot_group_scales(
                                slot=request.slot,
                                self_attn_scale=request.group_scales.get("self_attn", 1.0),
                                cross_attn_scale=request.group_scales.get("cross_attn", 1.0),
                                mlp_scale=request.group_scales.get("mlp", 1.0),
                            )
                            logger.info(f"[LoRA Load] Group scales applied successfully for slot {request.slot}")
                        except Exception as exc:
                            logger.warning(f"Failed to apply initial group scales for slot {request.slot}: {exc}")
                    else:
                        logger.info(f"[LoRA Load] No group_scales in request for slot {request.slot}")
                    return wrap_response({"message": result, "lora_path": request.lora_path, "slot": request.slot})
                else:
                    raise HTTPException(status_code=400, detail=result)

            # Basic mode: original PEFT-based loading
            adapter_name = request.adapter_name.strip() if isinstance(request.adapter_name, str) else None
            if adapter_name:
                result = handler.add_lora(request.lora_path, adapter_name=adapter_name)
            else:
                result = handler.load_lora(request.lora_path)

            if _is_success_message(result):
                # Atomically apply scale if provided
                if request.scale is not None:
                    try:
                        if adapter_name:
                            handler.set_lora_scale(adapter_name, request.scale)
                        else:
                            handler.set_lora_scale(request.scale)
                    except Exception as exc:
                        logger.warning(f"Failed to apply initial scale: {exc}")
                response_data: Dict[str, Any] = {"message": result, "lora_path": request.lora_path}
                if adapter_name:
                    response_data["adapter_name"] = adapter_name
                return wrap_response(response_data)
            raise HTTPException(status_code=400, detail=result)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load LoRA: {str(exc)}")

    @app.post("/v1/lora/unload")
    async def unload_lora_endpoint(request: Request, _: None = Depends(verify_api_key)):
        """Unload LoRA adapter and restore base model."""
        handler = _require_initialized_handler(app)
        try:
            # Check for slot param in body (advanced mode)
            slot = None
            try:
                body = await request.json()
                slot = body.get("slot")
            except Exception:
                pass

            if slot is not None:
                result = handler.unload_lora_slot(slot=int(slot))
            else:
                # If we have advanced slots loaded, unload all of them
                if hasattr(handler, '_adapter_slots') and handler._adapter_slots:
                    result = handler.unload_lora_slot(slot=None)
                else:
                    result = handler.unload_lora()

            if _is_success_message(result, allow_warning=True):
                return wrap_response({"message": result})
            raise HTTPException(status_code=400, detail=result)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to unload LoRA: {str(exc)}")

    @app.post("/v1/lora/toggle")
    async def toggle_lora_endpoint(request: ToggleLoRARequest, _: None = Depends(verify_api_key)):
        """Enable or disable LoRA adapter for inference."""
        handler = _require_initialized_handler(app)
        try:
            # If advanced adapter slots are loaded, use the advanced toggle
            if hasattr(handler, '_adapter_slots') and handler._adapter_slots:
                result = handler.set_use_lora_advanced(request.use_lora)
            else:
                result = handler.set_use_lora(request.use_lora)

            if _is_success_message(result):
                return wrap_response({"message": result, "use_lora": request.use_lora})
            return wrap_response(None, code=400, error=result)
        except Exception as exc:
            return wrap_response(None, code=500, error=f"Failed to toggle LoRA: {str(exc)}")

    @app.post("/v1/lora/scale")
    async def set_lora_scale_endpoint(request: SetLoRAScaleRequest, _: None = Depends(verify_api_key)):
        """Set LoRA adapter scale/strength (0.0-2.0)."""
        handler = _require_initialized_handler(app)
        try:
            # Advanced mode: slot-based scale
            if request.slot is not None:
                result = handler.set_lora_slot_scale(request.scale, request.slot)
                if _is_success_message(result):
                    return wrap_response({"message": result, "scale": request.scale, "slot": request.slot})
                else:
                    return wrap_response(None, code=400, error=result)

            # Basic mode: original PEFT-based scaling
            adapter_name = request.adapter_name.strip() if isinstance(request.adapter_name, str) else None
            if adapter_name:
                result = handler.set_lora_scale(adapter_name, request.scale)
            else:
                result = handler.set_lora_scale(request.scale)

            if _is_success_message(result, allow_warning=True):
                response_data: Dict[str, Any] = {"message": result, "scale": request.scale}
                if adapter_name:
                    response_data["adapter_name"] = adapter_name
                return wrap_response(response_data)
            return wrap_response(None, code=400, error=result)
        except Exception as exc:
            return wrap_response(None, code=500, error=f"Failed to set LoRA scale: {str(exc)}")

    @app.get("/v1/lora/status")
    async def get_lora_status_endpoint(_: None = Depends(verify_api_key)):
        """Get current LoRA/LoKr adapter state for the primary handler."""
        handler = _require_initialized_handler(app)
        status = handler.get_lora_status()
        # Build combined trigger_word from all slots (advanced) or fall back to single (basic)
        _adapter_slots: dict = getattr(handler, "_adapter_slots", {})
        if _adapter_slots:
            _seen = set()
            _words = []
            for _sid in sorted(_adapter_slots.keys()):
                _tw = _adapter_slots[_sid].get("trigger_word", "")
                if _tw and _tw not in _seen:
                    _words.append(_tw)
                    _seen.add(_tw)
            combined_trigger_word = ", ".join(_words)
            combined_tag_position = "prepend"  # multi-slot positions are handled per-slot at generation time
        else:
            combined_trigger_word = getattr(handler, "_adapter_trigger_word", "")
            combined_tag_position = getattr(handler, "_adapter_tag_position", "")

        result = {
            "lora_loaded": bool(status.get("loaded", getattr(handler, "lora_loaded", False))),
            "use_lora": bool(status.get("active", getattr(handler, "use_lora", False))),
            "lora_scale": float(status.get("scale", getattr(handler, "lora_scale", 1.0))),
            "adapter_type": getattr(handler, "_adapter_type", None),
            "trigger_word": combined_trigger_word,
            "tag_position": combined_tag_position,
            "scales": status.get("scales", {}),
            "active_adapter": status.get("active_adapter"),
            "adapters": status.get("adapters", []),
            "synthetic_default_mode": bool(status.get("synthetic_default_mode", False)),
            "merge_mode": bool(status.get("merge_mode", getattr(handler, "_adapter_merge_mode", False))),
        }
        # Include advanced adapter info if any slots are loaded
        if hasattr(handler, "get_advanced_lora_status"):
            advanced = handler.get_advanced_lora_status()
            result["advanced"] = advanced
        return wrap_response(result)

    @app.post("/v1/lora/merge-mode")
    async def set_merge_mode_endpoint(request: Request, _: None = Depends(verify_api_key)):
        """Toggle VRAM-optimized adapter merge mode.

        When enabled, adapters are baked into the base model weights instead
        of using PEFT runtime injection.  Zero extra VRAM for adapters, but
        scale changes trigger a re-merge (~3-5s).
        """
        handler = _require_initialized_handler(app)
        body = await request.json()
        enabled = body.get("enabled")
        if enabled is None:
            return wrap_response(None, code=400, error="Missing 'enabled' field")

        handler._adapter_merge_mode = bool(enabled)
        logger.info(f"Adapter merge mode set to: {handler._adapter_merge_mode}")
        return wrap_response({
            "merge_mode": handler._adapter_merge_mode,
            "message": f"Adapter merge mode {'enabled' if handler._adapter_merge_mode else 'disabled'}",
        })

    # ── File browser endpoint ────────────────────────────────────────

    @app.get("/v1/lora/list-files")
    async def list_lora_files_endpoint(folder: str = "", _: None = Depends(verify_api_key)):
        """List adapter files in the given directory."""
        if not folder:
            return wrap_response(None, code=400, error="Missing 'folder' query parameter")

        # Resolve relative paths against project root
        if not os.path.isabs(folder):
            folder = os.path.join(get_project_root(), folder)

        if not os.path.isdir(folder):
            return wrap_response(None, code=400, error=f"Directory not found: {folder}")

        ADAPTER_EXTENSIONS = {".safetensors", ".bin", ".pt"}
        files = []
        try:
            for entry in os.scandir(folder):
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in ADAPTER_EXTENSIONS:
                        stat = entry.stat()
                        adapter_type = "lokr" if "lokr" in entry.name.lower() else (
                            "lora" if "lora" in entry.name.lower() else "unknown"
                        )
                        files.append({
                            "name": entry.name,
                            "path": entry.path.replace("\\", "/"),
                            "size": stat.st_size,
                            "type": adapter_type,
                        })
                elif entry.is_dir():
                    # Check for adapter directories (contain adapter_config.json)
                    config_path = os.path.join(entry.path, "adapter_config.json")
                    if os.path.isfile(config_path):
                        files.append({
                            "name": entry.name,
                            "path": entry.path.replace("\\", "/"),
                            "size": 0,
                            "type": "peft_dir",
                        })
                    # Also check for safetensors files inside subdirs
                    for sub_entry in os.scandir(entry.path):
                        if sub_entry.is_file():
                            ext = os.path.splitext(sub_entry.name)[1].lower()
                            if ext in ADAPTER_EXTENSIONS:
                                stat = sub_entry.stat()
                                adapter_type = "lokr" if "lokr" in sub_entry.name.lower() else (
                                    "lora" if "lora" in sub_entry.name.lower() else "unknown"
                                )
                                files.append({
                                    "name": f"{entry.name}/{sub_entry.name}",
                                    "path": sub_entry.path.replace("\\", "/"),
                                    "size": stat.st_size,
                                    "type": adapter_type,
                                })
        except PermissionError:
            return wrap_response(None, code=403, error=f"Permission denied: {folder}")
        except Exception as exc:
            return wrap_response(None, code=500, error=f"Failed to scan folder: {str(exc)}")

        return wrap_response({"files": files, "folder": folder})

    # ── Advanced adapter endpoints ──────────────────────────────────

    @app.post("/v1/lora/group-scales")
    async def set_group_scales_endpoint(request: SetGroupScalesRequest, _: None = Depends(verify_api_key)):
        """Set per-module-group global scales for all adapter slots."""
        handler = _require_initialized_handler(app)
        try:
            result = handler.set_lora_group_scales(
                self_attn_scale=request.self_attn,
                cross_attn_scale=request.cross_attn,
                mlp_scale=request.mlp,
                cond_embed_scale=request.cond_embed,
            )
            if _is_success_message(result):
                return wrap_response({
                    "message": result,
                    "group_scales": {"self_attn": request.self_attn, "cross_attn": request.cross_attn, "mlp": request.mlp, "cond_embed": request.cond_embed},
                })
            return wrap_response(None, code=400, error=result)
        except Exception as exc:
            return wrap_response(None, code=500, error=f"Failed to set group scales: {str(exc)}")

    @app.post("/v1/lora/slot-group-scales")
    async def set_slot_group_scales_endpoint(request: SetSlotGroupScalesRequest, _: None = Depends(verify_api_key)):
        """Set per-group LoRA scales for a specific adapter slot."""
        handler = _require_initialized_handler(app)
        try:
            logger.info(f"[Slot Group Scales] Setting scales for slot {request.slot}: self_attn={request.self_attn}, cross_attn={request.cross_attn}, mlp={request.mlp}, cond_embed={request.cond_embed}")
            result = handler.set_slot_group_scales(
                slot=request.slot,
                self_attn_scale=request.self_attn,
                cross_attn_scale=request.cross_attn,
                mlp_scale=request.mlp,
                cond_embed_scale=request.cond_embed,
            )
            if _is_success_message(result):
                return wrap_response({
                    "message": result, "slot": request.slot,
                    "group_scales": {"self_attn": request.self_attn, "cross_attn": request.cross_attn, "mlp": request.mlp, "cond_embed": request.cond_embed},
                })
            return wrap_response(None, code=400, error=result)
        except Exception as exc:
            return wrap_response(None, code=500, error=f"Failed to set slot group scales: {str(exc)}")

    @app.post("/v1/lora/slot-layer-scales")
    async def set_slot_layer_scales_endpoint(request: SetSlotLayerScalesRequest, _: None = Depends(verify_api_key)):
        """Set per-layer LoRA scales for a specific adapter slot."""
        handler = _require_initialized_handler(app)
        try:
            result = handler.set_slot_layer_scales(slot=request.slot, layer_scales=request.layer_scales)
            if _is_success_message(result):
                return wrap_response({"message": result, "slot": request.slot, "layer_scales": request.layer_scales})
            return wrap_response(None, code=400, error=result)
        except Exception as exc:
            return wrap_response(None, code=500, error=f"Failed to set slot layer scales: {str(exc)}")

    @app.post("/v1/lora/slot-trigger-word")
    async def set_slot_trigger_word_endpoint(request: SetSlotTriggerWordRequest, _: None = Depends(verify_api_key)):
        """Set trigger word and placement for an adapter slot or basic adapter."""
        handler = _require_initialized_handler(app)
        try:
            tw = request.trigger_word.strip()
            tp = request.tag_position.strip().lower() if request.tag_position else "prepend"
            if tp not in ("prepend", "append", "replace"):
                tp = "prepend"

            if request.slot is not None:
                # Advanced mode: per-slot
                _adapter_slots: dict = getattr(handler, "_adapter_slots", {})
                if request.slot not in _adapter_slots:
                    return wrap_response(None, code=400, error=f"Slot {request.slot} not loaded")
                _adapter_slots[request.slot]["trigger_word"] = tw
                _adapter_slots[request.slot]["tag_position"] = tp
                logger.info(f"Slot {request.slot} trigger word set: '{tw}' (position: {tp})")
            else:
                # Basic mode: single adapter
                handler._adapter_trigger_word = tw
                handler._adapter_tag_position = tp
                logger.info(f"Basic adapter trigger word set: '{tw}' (position: {tp})")

            return wrap_response({
                "message": f"✅ Trigger word {'set' if tw else 'cleared'}",
                "trigger_word": tw,
                "tag_position": tp,
                "slot": request.slot,
            })
        except Exception as exc:
            return wrap_response(None, code=500, error=f"Failed to set trigger word: {str(exc)}")

    @app.post("/v1/lora/slot-layer-scale")
    async def set_slot_layer_scale_endpoint(request: SetSlotLayerScaleRequest, _: None = Depends(verify_api_key)):
        """Set scale for a single transformer layer on a specific adapter slot."""
        handler = _require_initialized_handler(app)
        try:
            result = handler.set_slot_layer_scale(slot=request.slot, layer=request.layer, scale=request.scale)
            if _is_success_message(result):
                return wrap_response({"message": result, "slot": request.slot, "layer": request.layer, "scale": request.scale})
            return wrap_response(None, code=400, error=result)
        except Exception as exc:
            return wrap_response(None, code=500, error=f"Failed to set slot layer scale: {str(exc)}")

    @app.post("/v1/lora/temporal-schedule")
    async def set_temporal_schedule_endpoint(request: SetTemporalScheduleRequest, _: None = Depends(verify_api_key)):
        """Set or clear a temporal adapter schedule for multi-singer switching."""
        handler = _require_initialized_handler(app)
        try:
            if request.clear or request.slot_segments is None:
                result = handler.set_temporal_schedule(None)
            else:
                from acestep.core.generation.handler.lora.temporal_adapter_schedule import (
                    AdapterSegment,
                    TemporalAdapterSchedule,
                )
                slot_segs = {}
                for sid_str, segs in request.slot_segments.items():
                    sid = int(sid_str)
                    slot_segs[sid] = [
                        AdapterSegment(
                            start=s.get("start", 0.0), end=s.get("end", 1.0), scale=s.get("scale", 1.0),
                            fade_in=s.get("fade_in", 0.0), fade_out=s.get("fade_out", 0.0),
                        )
                        for s in segs
                    ]
                schedule = TemporalAdapterSchedule(slot_segments=slot_segs)
                result = handler.set_temporal_schedule(schedule)

            if _is_success_message(result):
                return wrap_response({"message": result})
            return wrap_response(None, code=400, error=result)
        except Exception as exc:
            return wrap_response(None, code=500, error=f"Failed to set temporal schedule: {str(exc)}")

    @app.post("/v1/audio/diff")
    async def audio_diff_endpoint(request: AudioDiffRequest, _: None = Depends(verify_api_key)):
        """Compute audio difference between two tracks for layer ablation analysis."""

        def _resolve(p: str) -> str:
            project_root = get_project_root()
            if p.startswith("/audio/"):
                return os.path.join(project_root, "ace-step-ui", "server", "public", p.lstrip("/"))
            elif p.startswith("/v1/audio"):
                import urllib.parse as _urlparse
                parsed = _urlparse.urlparse(p)
                qs = _urlparse.parse_qs(parsed.query)
                if "path" in qs:
                    return qs["path"][0]
            elif not p.startswith("http") and not os.path.isabs(p):
                return os.path.join(project_root, p)
            return p

        reference_path = _resolve(request.reference_path)
        ablated_path = _resolve(request.ablated_path)

        if not os.path.isfile(reference_path):
            return wrap_response(None, code=400, error=f"Reference file not found: {reference_path}")
        if not os.path.isfile(ablated_path):
            return wrap_response(None, code=400, error=f"Ablated file not found: {ablated_path}")

        try:
            from acestep.core.generation.handler.lora.ablation_service import compute_audio_diff
            base = os.path.splitext(ablated_path)[0]
            output_path = f"{base}_diff.wav"
            result = compute_audio_diff(
                reference_path=reference_path, ablated_path=ablated_path,
                output_path=output_path, amplify=request.amplify,
            )
            return wrap_response(result)
        except Exception as exc:
            return wrap_response(None, code=500, error=f"Failed to compute audio diff: {str(exc)}")

    # ── Redmond Mode endpoints ─────────────────────────────────────

    @app.post("/v1/redmond/toggle")
    async def toggle_redmond_endpoint(request: RedmondToggleRequest, _: None = Depends(verify_api_key)):
        """Enable or disable Redmond Mode (DPO quality refinement)."""
        handler = _require_initialized_handler(app)
        try:
            from acestep.core.generation.handler.redmond_mode import toggle_redmond_mode
            result = toggle_redmond_mode(handler, request.enabled)
            if _is_success_message(result):
                return wrap_response({"message": result, "enabled": request.enabled})
            return wrap_response(None, code=400, error=result)
        except Exception as exc:
            return wrap_response(None, code=500, error=f"Failed to toggle Redmond Mode: {str(exc)}")

    @app.post("/v1/redmond/scale")
    async def set_redmond_scale_endpoint(request: RedmondScaleRequest, _: None = Depends(verify_api_key)):
        """Set Redmond Mode scale (0.0-2.0)."""
        handler = _require_initialized_handler(app)
        try:
            from acestep.core.generation.handler.redmond_mode import set_redmond_scale
            result = set_redmond_scale(handler, request.scale)
            if _is_success_message(result):
                return wrap_response({"message": result, "scale": request.scale})
            return wrap_response(None, code=400, error=result)
        except Exception as exc:
            return wrap_response(None, code=500, error=f"Failed to set Redmond scale: {str(exc)}")

    @app.get("/v1/redmond/status")
    async def get_redmond_status_endpoint(_: None = Depends(verify_api_key)):
        """Get current Redmond Mode state."""
        handler = _require_initialized_handler(app)
        from acestep.core.generation.handler.redmond_mode import get_redmond_status
        return wrap_response(get_redmond_status(handler))
