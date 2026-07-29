"""
routes/tts.py — Text-to-Speech endpoint
=========================================

``POST /tts`` accepts text + emotion + voice parameters and returns a
URL to the generated audio file.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from config import GENERATED_DIR

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tts"])


# ─────────────────────────────────────────────────────────────────────────────
#  Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    """Body for ``POST /tts``."""
    text: str = Field(..., min_length=1, max_length=2000, description="Text to synthesise.")
    emotion: str = Field("neutral", description="Emotion label for voice style.")
    voice_params: Optional[dict] = Field(
        None,
        description="Voice params from generate_voice_params() — speed, pitch, warmth.",
    )


class TTSResponse(BaseModel):
    """Response from ``POST /tts``."""
    audio_url: Optional[str] = Field(None, description="URL path to the generated audio file.")
    error: Optional[str] = Field(None, description="Error message if synthesis failed.")


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/tts", response_model=TTSResponse)
async def synthesize_speech(body: TTSRequest, request: Request) -> TTSResponse:
    """
    Synthesise speech from *text* using the configured TTS backend.

    Returns a relative URL to the generated audio file that can be
    fetched via the ``/generated`` static mount.
    """
    tts_service = getattr(request.app.state, "tts_service", None)

    if tts_service is None or not tts_service.available:
        return TTSResponse(
            audio_url=None,
            error="TTS service is not available. Use browser Web Speech API as fallback.",
        )

    try:
        filename = await tts_service.synthesize(
            text=body.text,
            emotion=body.emotion,
            voice_params=body.voice_params or {},
            output_dir=str(GENERATED_DIR),
        )
    except Exception as exc:
        logger.exception("TTS synthesis failed.")
        raise HTTPException(status_code=500, detail=f"TTS synthesis error: {exc}") from exc

    if filename is None:
        return TTSResponse(
            audio_url=None,
            error="TTS synthesis returned no output.",
        )

    # Periodic cleanup of old generated files
    try:
        tts_service.cleanup_old_files(str(GENERATED_DIR))
    except Exception:
        pass  # non-critical

    return TTSResponse(audio_url=f"/generated/{filename}")
