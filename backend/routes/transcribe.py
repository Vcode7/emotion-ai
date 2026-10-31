"""
routes/transcribe.py — Standalone transcription endpoint
=========================================================

``POST /transcribe`` accepts an audio upload and returns a text transcript.
The caller can select the STT provider via a form field:
    provider = "whisper"  → local OpenAI Whisper model
    provider = "groq"     → Groq Whisper API (cloud)

This endpoint is used by the Visual Chat pipeline on the frontend
to decouple speech-to-text from the LLM turn.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["transcribe"])


# ─────────────────────────────────────────────────────────────────────────────
#  Response schema
# ─────────────────────────────────────────────────────────────────────────────

class TranscribeResponse(BaseModel):
    """Response from ``POST /transcribe``."""
    text: str = ""
    provider: str = "whisper"
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    request: Request,
    audio: UploadFile = File(..., description="Audio file to transcribe (WebM/WAV/MP3)."),
    provider: str = Form("whisper", description="STT provider: 'whisper' or 'groq'."),
    groq_api_key: Optional[str] = Form(None, description="Groq API key (overrides server .env)."),
) -> TranscribeResponse:
    """
    Transcribe an uploaded audio file using the selected STT provider.

    Returns ``{"text": str, "provider": str}`` — or ``{"text": "", "error": str}``
    if transcription fails (the conversation pipeline continues regardless).
    """
    audio_bytes = await audio.read()
    filename = audio.filename or "recording.webm"

    if provider == "groq":
        groq_service = getattr(request.app.state, "groq_whisper_service", None)

        # Allow per-request API key override
        api_key = groq_api_key or None

        if groq_service is None:
            logger.warning("/transcribe: GroqWhisperService not initialised — falling back to local Whisper.")
        else:
            try:
                result = await groq_service.transcribe_bytes(
                    audio_bytes, filename, api_key_override=api_key
                )
                text = result.get("text", "").strip()
                if not text:
                    text = "(inaudible)"
                return TranscribeResponse(text=text, provider="groq")
            except Exception:
                logger.exception("/transcribe: Groq transcription failed — falling back to local Whisper.")

    # ── Local Whisper (default or fallback) ──────────────────────────────────
    whisper_service = getattr(request.app.state, "whisper_service", None)

    if whisper_service is None or not whisper_service.available:
        return TranscribeResponse(
            text="(transcription unavailable)",
            provider="whisper",
            error="Local Whisper model not available.",
        )

    try:
        import asyncio
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: whisper_service.transcribe_bytes(audio_bytes, filename),
        )
        text = result.get("text", "").strip()
        if not text:
            text = "(inaudible)"
        return TranscribeResponse(text=text, provider="whisper")
    except Exception:
        logger.exception("/transcribe: local Whisper transcription failed.")
        return TranscribeResponse(
            text="",
            provider="whisper",
            error="Transcription failed — please try again.",
        )
