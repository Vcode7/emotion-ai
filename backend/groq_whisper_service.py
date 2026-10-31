"""
groq_whisper_service.py — Groq Whisper API transcription service
=================================================================

Uses the Groq REST API (OpenAI-compatible endpoint) to transcribe audio.
This is a cloud-based alternative to the local OpenAI Whisper model.

Usage:
    svc = GroqWhisperService(api_key="gsk_...")
    result = await svc.transcribe_bytes(audio_bytes, filename="recording.webm")
    print(result["text"])
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from config import GROQ_API_KEY

logger = logging.getLogger(__name__)

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"


class GroqWhisperService:
    """
    Thin async wrapper around the Groq Whisper transcription API.

    Falls back gracefully if the API key is missing or the request fails.

    Attributes:
        api_key:   The Groq API key (from .env or passed explicitly).
        available: ``True`` if an API key is configured.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        from config import get_groq_api_keys

        keys = get_groq_api_keys(api_key)
        self.api_key: str = keys[0] if keys else ""
        self.available: bool = bool(self.api_key)

        if self.available:
            logger.info("GroqWhisperService ready (model: %s, %d keys configured).", GROQ_WHISPER_MODEL, len(keys))
        else:
            logger.warning(
                "GroqWhisperService: no API key — Groq transcription will be unavailable."
            )

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "recording.webm",
        api_key_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe raw audio bytes using the Groq Whisper API.
        Automatically rotates through fallback keys if rate limits are hit.

        Args:
            audio_bytes:     Raw audio data (WebM, WAV, MP3, etc.).
            filename:        Hint for the MIME type detection (e.g. 'recording.webm').
            api_key_override: Use this key instead of the one set at init time.

        Returns:
            ``{"text": str, "segments": [...]}`` — segments carry
            ``no_speech_prob`` / ``avg_logprob`` for the speech gate.
            On failure the dict also has an ``"error"`` key so callers can
            fall back to local Whisper.
        """
        from config import get_groq_api_keys

        keys_to_try = get_groq_api_keys(api_key_override or self.api_key)
        if not keys_to_try:
            logger.warning("GroqWhisperService.transcribe_bytes: no API key available.")
            return {"text": "", "segments": [], "error": "no_api_key"}

        for idx, key in enumerate(keys_to_try):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        GROQ_TRANSCRIPTION_URL,
                        headers={"Authorization": f"Bearer {key}"},
                        files={
                            "file": (filename, audio_bytes, _mime_from_filename(filename)),
                        },
                        data={
                            "model": GROQ_WHISPER_MODEL,
                            "language": "en",
                            "response_format": "verbose_json",
                        },
                    )

                    # Only the status code signals a rate limit — the body of a
                    # successful response is the user's transcript.
                    if response.status_code == 429:
                        has_next = idx + 1 < len(keys_to_try)
                        if has_next:
                            logger.warning(
                                "[Groq Whisper STT] Rate limit hit on key %s... (HTTP 429). Switching to fallback key #%d...",
                                key[:10], idx + 1
                            )
                            continue
                        else:
                            logger.error(
                                "[Groq Whisper STT] Rate limit hit on key %s... and all %d Groq fallback keys were exhausted.",
                                key[:10], len(keys_to_try)
                            )
                            return {"text": "", "segments": [], "error": "rate_limited"}

                    response.raise_for_status()
                    data = response.json()
                    text = data.get("text", "").strip()
                    segments = [
                        {
                            "start": float(seg.get("start", 0.0)),
                            "end":   float(seg.get("end", 0.0)),
                            "text":  str(seg.get("text", "")).strip(),
                            "no_speech_prob": float(seg.get("no_speech_prob", 0.0)),
                            "avg_logprob":    float(seg.get("avg_logprob", 0.0)),
                        }
                        for seg in (data.get("segments") or [])
                        if isinstance(seg, dict)
                    ]
                    if idx > 0:
                        logger.info("[Groq Whisper STT] Fallback key #%d succeeded.", idx)
                    logger.debug("Groq Whisper transcript: %r", text[:80])
                    return {"text": text, "segments": segments}

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    has_next = idx + 1 < len(keys_to_try)
                    if has_next:
                        logger.warning(
                            "[Groq Whisper STT] Rate limit 429 on key %s... Switching to fallback key #%d...",
                            key[:10], idx + 1
                        )
                        continue
                logger.error(
                    "Groq Whisper API error %s: %s",
                    exc.response.status_code,
                    exc.response.text[:200],
                )
                return {"text": "", "segments": [], "error": f"http_{exc.response.status_code}"}
            except Exception as exc:
                logger.exception("GroqWhisperService.transcribe_bytes failed.")
                return {"text": "", "segments": [], "error": type(exc).__name__}

        return {"text": "", "segments": [], "error": "rate_limited"}


    async def transcribe_file(
        self,
        audio_path: str,
        api_key_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper: read a file from disk and transcribe it.

        Args:
            audio_path:      Path to the audio file on disk.
            api_key_override: Optional key override.

        Returns:
            ``{"text": str, "segments": []}``
        """
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            import os
            filename = os.path.basename(audio_path)
            return await self.transcribe_bytes(audio_bytes, filename, api_key_override)
        except OSError:
            logger.exception("GroqWhisperService: failed to read file '%s'.", audio_path)
            return {"text": "", "segments": [], "error": "file_read_failed"}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mime_from_filename(filename: str) -> str:
    """Return a best-guess MIME type from the file extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {
        "webm": "audio/webm",
        "wav":  "audio/wav",
        "mp3":  "audio/mpeg",
        "mp4":  "audio/mp4",
        "ogg":  "audio/ogg",
        "flac": "audio/flac",
        "m4a":  "audio/mp4",
    }.get(ext, "audio/webm")
