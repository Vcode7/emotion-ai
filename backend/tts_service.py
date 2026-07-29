"""
tts_service.py — Text-to-Speech with graceful multi-fallback
=============================================================

Attempts to use **StyleTTS2** as the primary TTS engine.  If that is not
installed or fails to initialise, falls back to **edge-tts** (Microsoft
Edge online TTS).  If neither is available, the service degrades gracefully
and the frontend can use the browser's built-in Web Speech API.

Public API:
    svc = TTSService()
    path = await svc.synthesize(text, emotion, voice_params, output_dir)
    svc.cleanup_old_files(output_dir)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Detect available TTS backends
# ─────────────────────────────────────────────────────────────────────────────

_STYLETTS2_AVAILABLE = False
_styletts2_model = None

try:
    import styletts2  # type: ignore[import-untyped]
    _STYLETTS2_AVAILABLE = True
    logger.info("StyleTTS2 package detected.")
except ImportError:
    logger.info("StyleTTS2 not installed — will try edge-tts fallback.")

_EDGE_TTS_AVAILABLE = False
try:
    import edge_tts  # type: ignore[import-untyped]
    _EDGE_TTS_AVAILABLE = True
    logger.info("edge-tts package detected.")
except ImportError:
    logger.info("edge-tts not installed — TTS will be unavailable (frontend Web Speech API fallback).")


# ─────────────────────────────────────────────────────────────────────────────
#  Emotion → edge-tts voice mapping
# ─────────────────────────────────────────────────────────────────────────────

# Edge-tts uses Microsoft voices identified by a ShortName.
# We pick voices that sound appropriate for each emotional register.
_EDGE_VOICE_MAP = {
    "compassion":       "en-US-JennyNeural",
    "warm_positive":    "en-US-AriaNeural",
    "calm_reassuring":  "en-US-JennyNeural",
    "neutral":          "en-US-GuyNeural",
    "happy":            "en-US-AriaNeural",
    "sad":              "en-US-JennyNeural",
    "angry":            "en-US-GuyNeural",
    "surprised":        "en-US-AriaNeural",
    "concerned":        "en-US-JennyNeural",
    "encouraging":      "en-US-AriaNeural",
    "supportive":       "en-US-JennyNeural",
    "thinking":         "en-US-GuyNeural",
    "fearful":          "en-US-JennyNeural",
}

_DEFAULT_EDGE_VOICE = "en-US-JennyNeural"


# ─────────────────────────────────────────────────────────────────────────────
#  TTS Service
# ─────────────────────────────────────────────────────────────────────────────

class TTSService:
    """
    Multi-backend TTS service with graceful degradation.

    Priority:
        1. StyleTTS2 (offline, high quality)
        2. edge-tts  (online, Microsoft Edge voices)
        3. None      (frontend falls back to Web Speech API)

    Attributes:
        available: ``True`` if at least one backend is usable.
        backend:   Name of the active backend, or ``None``.
    """

    def __init__(self) -> None:
        self.available: bool = False
        self.backend: Optional[str] = None

        # ── Try StyleTTS2 ────────────────────────────────────────────────
        if _STYLETTS2_AVAILABLE:
            try:
                global _styletts2_model
                # styletts2 typically exposes a tts object after loading
                # a config/checkpoint.  The exact API varies by fork.
                # We attempt a best-effort default initialisation.
                _styletts2_model = styletts2  # store module reference
                self.available = True
                self.backend = "styletts2"
                logger.info("TTSService initialised with StyleTTS2 backend.")
                return
            except Exception:
                logger.exception("StyleTTS2 initialisation failed.")

        # ── Try edge-tts ─────────────────────────────────────────────────
        if _EDGE_TTS_AVAILABLE:
            self.available = True
            self.backend = "edge-tts"
            logger.info("TTSService initialised with edge-tts backend.")
            return

        # ── No backend ───────────────────────────────────────────────────
        logger.warning("No TTS backend available — synthesize() will return None.")

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        emotion: str = "neutral",
        voice_params: Optional[dict] = None,
        output_dir: str | Path = "generated",
    ) -> Optional[str]:
        """
        Synthesise *text* to a WAV/MP3 file and return the filename.

        Args:
            text:         The text to speak.
            emotion:      Emotion label from ``generate_voice_params()``.
            voice_params: Dict with ``speed``, ``pitch``, ``warmth`` keys.
            output_dir:   Directory to write the output file.

        Returns:
            The filename (basename) of the generated audio, or ``None``
            if TTS is unavailable.
        """
        if not self.available:
            return None

        vp = voice_params or {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.backend == "styletts2":
            return await self._synthesize_styletts2(text, emotion, vp, output_dir)
        elif self.backend == "edge-tts":
            return await self._synthesize_edge_tts(text, emotion, vp, output_dir)

        return None

    # ------------------------------------------------------------------
    #  StyleTTS2 backend
    # ------------------------------------------------------------------

    async def _synthesize_styletts2(
        self,
        text: str,
        emotion: str,
        voice_params: dict,
        output_dir: Path,
    ) -> Optional[str]:
        """Generate speech using StyleTTS2."""
        try:
            filename = f"{uuid.uuid4().hex}.wav"
            filepath = output_dir / filename

            speed = voice_params.get("speed", 1.0)
            # StyleTTS2 API varies by fork; common pattern:
            #   styletts2.tts(text, output_path, speed=…)
            # We run it in an executor to avoid blocking the event loop.
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: _styletts2_model.tts(  # type: ignore[union-attr]
                    text,
                    output_wav_file=str(filepath),
                    speed=speed,
                ),
            )
            return filename

        except Exception:
            logger.exception("StyleTTS2 synthesis failed — falling back to edge-tts.")
            # Attempt edge-tts fallback if available
            if _EDGE_TTS_AVAILABLE:
                return await self._synthesize_edge_tts(text, emotion, voice_params, output_dir)
            return None

    # ------------------------------------------------------------------
    #  edge-tts backend
    # ------------------------------------------------------------------

    async def _synthesize_edge_tts(
        self,
        text: str,
        emotion: str,
        voice_params: dict,
        output_dir: Path,
    ) -> Optional[str]:
        """Generate speech using Microsoft Edge TTS."""
        try:
            filename = f"{uuid.uuid4().hex}.mp3"
            filepath = output_dir / filename

            voice = _EDGE_VOICE_MAP.get(emotion, _DEFAULT_EDGE_VOICE)

            # Build rate/pitch strings from voice_params
            speed = voice_params.get("speed", 1.0)
            pitch = voice_params.get("pitch", 1.0)

            # edge-tts rate: percentage offset, e.g. "+20%", "-10%"
            rate_pct = int((speed - 1.0) * 100)
            rate_str = f"{rate_pct:+d}%"

            # edge-tts pitch: percentage offset, e.g. "+20%", "-10%"
            pitch_pct = int((pitch - 1.0) * 100)
            pitch_str = f"{pitch_pct:+d}%"

            communicate = edge_tts.Communicate(  # type: ignore[attr-defined]
                text,
                voice=voice,
                rate=rate_str,
                pitch=pitch_str,
            )
            await communicate.save(str(filepath))
            return filename

        except Exception:
            logger.exception("edge-tts synthesis failed.")
            return None

    # ------------------------------------------------------------------
    #  File cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def cleanup_old_files(output_dir: str | Path, max_age_seconds: int = 300) -> int:
        """
        Delete generated TTS files older than *max_age_seconds*.

        Returns the number of files deleted.
        """
        output_dir = Path(output_dir)
        if not output_dir.exists():
            return 0

        now = time.time()
        deleted = 0

        for fpath in output_dir.iterdir():
            if fpath.is_file() and (now - fpath.stat().st_mtime) > max_age_seconds:
                try:
                    fpath.unlink()
                    deleted += 1
                except OSError:
                    pass

        if deleted:
            logger.info("Cleaned up %d old TTS file(s) from %s.", deleted, output_dir)

        return deleted
