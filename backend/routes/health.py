"""
routes/health.py — Health-check endpoint
==========================================

Provides a single ``GET /health`` endpoint that reports the runtime
status of all backend services.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(request: Request) -> dict:
    """
    Return the current health status of the backend.

    Response body:
        - ``status``        — always ``"ok"`` if this endpoint responds.
        - ``model_loaded``  — whether the DMESR model loaded successfully.
        - ``device``        — PyTorch compute device string.
        - ``sessions``      — number of active inference sessions.
        - ``tts_available`` — whether at least one TTS backend is active.
    """
    inference_manager = getattr(request.app.state, "inference_manager", None)
    tts_service = getattr(request.app.state, "tts_service", None)

    return {
        "status": "ok",
        "model_loaded": inference_manager.model_loaded if inference_manager else False,
        "device": inference_manager.device if inference_manager else "unknown",
        "sessions": inference_manager.session_count if inference_manager else 0,
        "tts_available": tts_service.available if tts_service else False,
    }
