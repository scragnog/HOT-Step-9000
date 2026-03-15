"""LM LoRA HTTP routes for merge-based adapter lifecycle on the 5Hz LLM.

Separate from the DiT LoRA routes — these operate on the language model,
not the diffusion transformer. Adapters are merged into the base model
weights and loaded via vLLM for full-speed generation.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from acestep.llm_inference import LLMHandler


# ── Request models ──────────────────────────────────────────────────────

class LoadLmLoraRequest(BaseModel):
    lm_lora_path: str = Field(..., description="Path to PEFT adapter directory")
    scale: float = Field(1.0, ge=0.0, le=10.0, description="LoRA scale (baked at merge time)")


class SetLmLoraScaleRequest(BaseModel):
    scale: float = Field(..., ge=0.0, le=10.0, description="LM LoRA scale (0.0-10.0)")


# ── Helpers ─────────────────────────────────────────────────────────────

def _require_llm_handler(app: FastAPI) -> LLMHandler:
    """Return initialized LLM handler or raise HTTP 500."""
    handler: LLMHandler = getattr(app.state, "llm_handler", None)
    if handler is None:
        raise HTTPException(status_code=500, detail="LLM handler not initialized")
    return handler


_SUCCESS = "\u2705"
_WARNING = "\u26a0\ufe0f"


def _is_ok(result: str) -> bool:
    return result.startswith(_SUCCESS) or result.startswith(_WARNING)


# ── Route registration ─────────────────────────────────────────────────

def register_lm_lora_routes(
    app: FastAPI,
    verify_api_key: Callable[..., Any],
    wrap_response: Callable[..., Dict[str, Any]],
) -> None:
    """Register LM LoRA (PEFT) lifecycle endpoints."""

    @app.post("/v1/lora/lm-load")
    async def load_lm_lora(request: LoadLmLoraRequest, _: None = Depends(verify_api_key)):
        """Load a PEFT LoRA adapter by merging into the base model.

        Merges the adapter at the specified scale, saves to a temp checkpoint,
        and reinitializes vLLM from the merged model. Takes ~2 min on first load
        (cached for subsequent loads at the same scale).
        """
        handler = _require_llm_handler(app)
        try:
            result = handler.load_lm_lora(request.lm_lora_path, request.scale)
            if _is_ok(result):
                return wrap_response({
                    "message": result,
                    "lm_lora_path": request.lm_lora_path,
                })
            raise HTTPException(status_code=400, detail=result)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load LM LoRA: {str(exc)}")

    @app.post("/v1/lora/lm-unload")
    async def unload_lm_lora(_: None = Depends(verify_api_key)):
        """Unload the PEFT LoRA adapter and restore the base LLM."""
        handler = _require_llm_handler(app)
        try:
            result = handler.unload_lm_lora()
            if _is_ok(result):
                return wrap_response({"message": result})
            raise HTTPException(status_code=400, detail=result)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to unload LM LoRA: {str(exc)}")

    @app.post("/v1/lora/lm-scale")
    async def set_lm_lora_scale(request: SetLmLoraScaleRequest, _: None = Depends(verify_api_key)):
        """Change the LM LoRA scale. Triggers a full re-merge + reinit (~2 min)."""
        handler = _require_llm_handler(app)
        try:
            result = handler.set_lm_lora_scale(request.scale)
            if _is_ok(result):
                return wrap_response({"message": result, "scale": request.scale})
            raise HTTPException(status_code=400, detail=result)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to set LM LoRA scale: {str(exc)}")

    @app.get("/v1/lora/lm-status")
    async def get_lm_lora_status(_: None = Depends(verify_api_key)):
        """Get current LM LoRA adapter state."""
        handler = _require_llm_handler(app)
        return wrap_response(handler.get_lm_lora_status())
