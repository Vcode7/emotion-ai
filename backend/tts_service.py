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
#  Emotion → edge-tts voice mapping  (FEMALE ONLY — Avatars Sky & Sky2 are female)
# ─────────────────────────────────────────────────────────────────────────────

# All voices are female Microsoft Edge neural voices.
# en-US-JennyNeural  — warm, natural, balanced conversational female voice (neutral, thinking, concerned, supportive)
# en-US-EmmaNeural   — bright, cheerful, lively, upbeat female voice (happy, warm_positive, encouraging)
# en-US-AvaNeural    — gentle, soft, comforting, empathetic female voice (compassion, calm_reassuring, empathetic, sad)
# en-US-AriaNeural   — expressive, dynamic, emotionally rich female voice (surprised, angry, fearful, fear, disgust)

_FEMALE_EDGE_VOICES = {
    "en-US-JennyNeural",
    "en-US-AvaNeural",
    "en-US-EmmaNeural",
    "en-US-AriaNeural",
    "en-US-MichelleNeural",
    "en-US-AnaNeural",
}

_EDGE_VOICE_MAP = {
    # Warm / Gentle / Empathetic female voices (AvaNeural)
    "compassion":       "en-US-AvaNeural",
    "calm_reassuring":  "en-US-AvaNeural",
    "empathetic":       "en-US-AvaNeural",
    "sad":              "en-US-AvaNeural",
    "sadness":          "en-US-AvaNeural",
    "relief":           "en-US-AvaNeural",
    "calm":             "en-US-AvaNeural",

    # Bright / Cheerful / Upbeat female voices (EmmaNeural)
    "happy":            "en-US-EmmaNeural",
    "happiness":        "en-US-EmmaNeural",
    "warm_positive":    "en-US-EmmaNeural",
    "encouraging":      "en-US-EmmaNeural",
    "joy":              "en-US-EmmaNeural",
    "excited":          "en-US-EmmaNeural",

    # Expressive / Dynamic / Intense female voices (AriaNeural)
    "surprised":        "en-US-AriaNeural",
    "surprise":         "en-US-AriaNeural",
    "angry":            "en-US-AriaNeural",
    "anger":            "en-US-AriaNeural",
    "fearful":          "en-US-AriaNeural",
    "fear":             "en-US-AriaNeural",
    "disgust":          "en-US-AriaNeural",
    "contempt":         "en-US-AriaNeural",
    "anxious":          "en-US-AriaNeural",

    # Natural / Balanced / Thoughtful female voices (JennyNeural)
    "neutral":          "en-US-JennyNeural",
    "thinking":         "en-US-JennyNeural",
    "concerned":        "en-US-JennyNeural",
    "supportive":       "en-US-JennyNeural",
}

_DEFAULT_EDGE_VOICE = "en-US-JennyNeural"  # female fallback for any unknown emotion


# ─────────────────────────────────────────────────────────────────────────────
#  TTS Service
# ─────────────────────────────────────────────────────────────────────────────

class TTSService:
    """
    Multi-backend TTS service with graceful degradation.

    Priority:
        1. StyleTTS2 (offline, high quality)
        2. edge-tts  (online, Microsoft Edge voices — FEMALE ONLY)
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
            logger.info("TTSService initialised with edge-tts backend (FEMALE voices only).")
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

        Note:
            Avatars (Sky & Sky2) are FEMALE — all voices are strictly female regardless of emotion.
        """
        if not self.available:
            return None

        vp = voice_params or {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        emotion_lower = (emotion or "neutral").lower().strip()
        voice_selected = _EDGE_VOICE_MAP.get(emotion_lower, _DEFAULT_EDGE_VOICE)
        if voice_selected not in _FEMALE_EDGE_VOICES:
            voice_selected = _DEFAULT_EDGE_VOICE

        logger.info("TTS synthesize: emotion=%r → voice=%s (female)",
                    emotion_lower, voice_selected)

        if self.backend == "styletts2":
            return await self._synthesize_styletts2(text, emotion_lower, vp, output_dir)
        elif self.backend == "edge-tts":
            return await self._synthesize_edge_tts(text, emotion_lower, vp, output_dir)

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
        """Generate speech using Microsoft Edge TTS (FEMALE voices only)."""
        try:
            filename = f"{uuid.uuid4().hex}.mp3"
            filepath = output_dir / filename

            # Always pick a FEMALE voice — models are female avatars (Sky & Sky2)
            voice = _EDGE_VOICE_MAP.get(emotion, _DEFAULT_EDGE_VOICE)
            if voice not in _FEMALE_EDGE_VOICES:
                voice = _DEFAULT_EDGE_VOICE
            logger.info("edge-tts: emotion=%r → voice=%s (female)", emotion, voice)

            # Build rate/pitch strings from voice_params
            speed = voice_params.get("speed", 1.0)
            pitch = voice_params.get("pitch", 1.0)

            # edge-tts rate: percentage offset, e.g. "+20%", "-10%"
            rate_pct = int((speed - 1.0) * 100)
            rate_str = f"{rate_pct:+d}%"

            # edge-tts pitch: offset in Hz, e.g. "+5Hz", "-5Hz" (edge-tts requires r"^[+-]\d+Hz$")
            pitch_hz = int((pitch - 1.0) * 50)
            pitch_str = f"{pitch_hz:+d}Hz"

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
