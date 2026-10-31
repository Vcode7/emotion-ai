"""
whisper_service.py — OpenAI Whisper ASR wrapper
=================================================

Provides a thin, error-resilient interface around the ``whisper`` package
for transcribing uploaded audio files.

Usage:
    svc = WhisperService()          # loads the model once
    result = svc.transcribe("/path/to/file.wav")
    print(result["text"])           # full transcript
    print(result["segments"])       # [{start, end, text}, …]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import whisper

from config import DEVICE, WHISPER_MODEL_SIZE

logger = logging.getLogger(__name__)


class WhisperService:
    """
    Wraps the OpenAI Whisper model for speech-to-text transcription.

    Attributes:
        model:     The loaded ``whisper.Whisper`` model instance.
        available: ``True`` if the model loaded without error.
    """

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE, device: str = DEVICE) -> None:
        """
        Load the Whisper model.

        Args:
            model_size: One of 'tiny', 'base', 'small', 'medium', 'large'.
            device:     Compute device ('cpu' or 'cuda').
        """
        self.available: bool = False
        self.model: whisper.Whisper | None = None

        try:
            logger.info("Loading Whisper model '%s' on '%s' …", model_size, device)
            self.model = whisper.load_model(model_size, device=device)
            self.available = True
            logger.info("Whisper model loaded successfully.")
        except Exception:
            logger.exception("Failed to load Whisper model — ASR will be unavailable.")

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def transcribe_bytes(self, audio_bytes: bytes, filename: str = "recording.webm") -> Dict[str, Any]:
        """
        Transcribe raw audio bytes by saving them to a temporary file, then
        transcribing and cleaning up.

        Args:
            audio_bytes: Raw audio data (WebM, WAV, MP3, etc.).
            filename:    Hint used for the temporary filename extension.

        Returns:
            Same schema as ``transcribe()``: ``{text, segments}``.
        """
        import tempfile, os, pathlib
        ext = pathlib.Path(filename).suffix or ".webm"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            return self.transcribe(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe an audio file and return the text plus segment-level
        timestamps.

        Args:
            audio_path: Absolute path to the audio file (WAV, MP3, etc.).

        Returns:
            A dict with:
                ``text``     — full transcript string.
                ``segments`` — list of ``{start: float, end: float, text: str}``.

        If the model is unavailable or an error occurs, returns an empty
        transcript with a descriptive ``text`` field.
        """
        if not self.available or self.model is None:
            logger.warning("Whisper model not available — returning empty transcript.")
            return {
                "text": "",
                "segments": [],
            }

        try:
            result = self.model.transcribe(
                audio_path,
                language="en",       # force English for speed; remove to auto-detect
                fp16=(DEVICE == "cuda"),
                # Utterances are short and independent; conditioning on previous
                # text makes Whisper loop on hallucinations.
                condition_on_previous_text=False,
            )

            # no_speech_prob / avg_logprob feed the speech gate (speech_gate.py).
            segments: List[Dict[str, Any]] = [
                {
                    "start": float(seg["start"]),
                    "end":   float(seg["end"]),
                    "text":  seg["text"].strip(),
                    "no_speech_prob": float(seg.get("no_speech_prob", 0.0)),
                    "avg_logprob":    float(seg.get("avg_logprob", 0.0)),
                }
                for seg in result.get("segments", [])
            ]

            return {
                "text":     result.get("text", "").strip(),
                "segments": segments,
            }

        except Exception:
            logger.exception("Whisper transcription failed for '%s'.", audio_path)
            return {
                "text": "",
                "segments": [],
            }
