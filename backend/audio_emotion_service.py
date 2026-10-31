"""
audio_emotion_service.py — Speech Emotion Recognition via Wav2Vec2 SUPERB-ER
=============================================================================

Uses the pretrained ``superb/wav2vec2-base-superb-er`` model for Speech
Emotion Recognition (SER).

Why this model instead of ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition?
  - The ehcalabres model was fine-tuned ONLY on RAVDESS (acted, studio speech).
    On real conversational audio it is heavily biased toward "angry/disgust".
  - superb/wav2vec2-base-superb-er is trained on the SUPERB benchmark which
    includes IEMOCAP (natural, spontaneous conversational speech) — much better
    calibrated for real-world use.

Model details:
    Checkpoint  : superb/wav2vec2-base-superb-er
    Task        : Sequence classification (speech → emotion)
    Input       : 16 kHz mono float32 waveform (torch.Tensor or numpy array)
    Output      : Top emotion + confidence + full probability dict
    Labels      : neu (neutral), hap (happy), ang (angry), sad (sad)
    Accuracy    : ~65% on IEMOCAP 4-class held-out set
    VRAM        : ~400 MB GPU / ~600 MB CPU RAM for wav2vec2-base
    Sample rate : 16 000 Hz (must resample before passing)

Confidence gating:
    If the top-1 confidence is below CONFIDENCE_GATE (35%), the result falls
    back to "neutral". This prevents low-quality or very short audio from
    producing spurious emotion labels.

Lazy loading: the model is loaded at construction time.

IMPORTANT: This service runs INDEPENDENTLY of Whisper/Groq transcription.
The same audio buffer feeds both the transcription path and this SER path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Model configuration
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_NAME  = "superb/wav2vec2-base-superb-er"
_SAMPLE_RATE = 16_000

# Below this confidence the model is considered uncertain → return neutral
CONFIDENCE_GATE = 0.35

# SUPERB-ER label → standard pipeline vocabulary
# The model uses short codes: neu, hap, ang, sad
_SUPERB_TO_STANDARD: Dict[str, str] = {
    "neu":  "neutral",
    "hap":  "happy",
    "ang":  "angry",
    "sad":  "sad",
    # safety net for any extra labels the model config might list
    "neutral":   "neutral",
    "happy":     "happy",
    "angry":     "angry",
    "sad":       "sad",
    "excited":   "happy",
    "frustrated": "angry",
}


class AudioEmotionService:
    """
    Service for Speech Emotion Recognition on raw 16 kHz audio waveforms.

    Uses ``superb/wav2vec2-base-superb-er`` trained on the SUPERB benchmark
    (IEMOCAP conversational speech) for balanced, real-world emotion detection.

    Attributes:
        available (bool): True when the model loaded successfully.
        model_name (str): HuggingFace model checkpoint in use.
    """

    def __init__(self) -> None:
        self._model     = None
        self._processor = None
        self._id2label: Dict[int, str] = {}
        self.available  = False
        self.model_name = _MODEL_NAME
        self._device    = "cuda" if torch.cuda.is_available() else "cpu"
        self._load()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the Wav2Vec2 SER model from HuggingFace."""
        try:
            from transformers import (  # noqa: PLC0415
                AutoFeatureExtractor,
                AutoModelForAudioClassification,
            )

            logger.info("=" * 60)
            logger.info("[AudioEmotion] Loading model: %s", _MODEL_NAME)
            logger.info("[AudioEmotion] Device: %s", self._device)
            logger.info("[AudioEmotion] Confidence gate: %.0f%%", CONFIDENCE_GATE * 100)

            self._processor = AutoFeatureExtractor.from_pretrained(
                _MODEL_NAME, trust_remote_code=False
            )
            self._model = AutoModelForAudioClassification.from_pretrained(
                _MODEL_NAME, trust_remote_code=False
            ).to(self._device)
            self._model.eval()

            # Build id→label lookup from model config
            self._id2label = {
                int(k): v.lower()
                for k, v in self._model.config.id2label.items()
            }

            label_list = list(self._id2label.values())
            self.available = True

            logger.info("[AudioEmotion] ✓ Model loaded successfully!")
            logger.info("[AudioEmotion]   Checkpoint : %s", _MODEL_NAME)
            logger.info("[AudioEmotion]   Labels (%d): %s", len(label_list), label_list)
            logger.info("[AudioEmotion]   Device     : %s", self._device)
            logger.info("[AudioEmotion]   Gate       : top-1 < %.0f%% → neutral", CONFIDENCE_GATE * 100)
            logger.info("=" * 60)

        except Exception as exc:
            logger.error("=" * 60)
            logger.error("[AudioEmotion] ✗ FAILED to load model: %s", _MODEL_NAME)
            logger.error("[AudioEmotion]   Error: %s", exc)
            logger.error("[AudioEmotion]   Audio emotion will return neutral for all turns.")
            logger.error("=" * 60)
            self.available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, waveform: "torch.Tensor | np.ndarray") -> Dict[str, Any]:
        """
        Classify the emotion in a raw audio waveform.

        Parameters
        ----------
        waveform : torch.Tensor or np.ndarray
            1-D float32 samples at 16 000 Hz.  Must be at least 0.5 s long
            (8 000 samples).  Longer clips are truncated to 30 s.

        Returns
        -------
        dict with keys:
            emotion       (str)   — mapped standard emotion label
            raw_emotion   (str)   — original model label (e.g. "neu", "hap")
            confidence    (float) — probability of the top label [0.0, 1.0]
            probabilities (dict)  — {raw_label: score}
            gated         (bool)  — True when confidence gate suppressed result
            available     (bool)  — True when model is loaded
        """
        if not self.available:
            return _unavailable("model_not_loaded")

        try:
            # ── Convert to numpy ──────────────────────────────────────────
            if isinstance(waveform, torch.Tensor):
                audio_np = waveform.cpu().float().numpy()
            else:
                audio_np = np.asarray(waveform, dtype=np.float32)

            # Flatten to 1-D
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()

            # Gate: need at least 0.5 s
            if len(audio_np) < _SAMPLE_RATE // 2:
                logger.debug(
                    "[AudioEmotion] Audio too short (%d samples < 8000) — returning neutral",
                    len(audio_np),
                )
                return _unavailable("too_short")

            # Truncate to 30 s to avoid OOM on very long clips
            max_samples = 30 * _SAMPLE_RATE
            if len(audio_np) > max_samples:
                audio_np = audio_np[:max_samples]

            # ── Feature extraction ────────────────────────────────────────
            inputs = self._processor(
                audio_np,
                sampling_rate=_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # ── Inference ─────────────────────────────────────────────────
            with torch.no_grad():
                logits = self._model(**inputs).logits   # (1, num_labels)

            # Softmax → probabilities
            probs    = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            top_idx  = int(np.argmax(probs))
            top_raw  = self._id2label.get(top_idx, "neu")
            confidence = round(float(probs[top_idx]), 4)

            # Map raw label → standard vocabulary
            mapped = _SUPERB_TO_STANDARD.get(top_raw, top_raw)

            probabilities = {
                self._id2label.get(i, str(i)): round(float(p), 4)
                for i, p in enumerate(probs)
            }

            # ── Confidence gating ─────────────────────────────────────────
            gated = confidence < CONFIDENCE_GATE
            if gated:
                logger.debug(
                    "[AudioEmotion] Gated: raw=%s (%.1f%%) below gate %.0f%% → neutral",
                    top_raw, confidence * 100, CONFIDENCE_GATE * 100,
                )
                return {
                    "emotion":       "neutral",
                    "raw_emotion":   top_raw,
                    "confidence":    confidence,
                    "probabilities": probabilities,
                    "gated":         True,
                    "available":     True,
                }

            logger.debug(
                "[AudioEmotion] raw=%s (%.1f%%) → mapped=%s | probs=%s",
                top_raw,
                confidence * 100,
                mapped,
                {k: f"{v:.2f}" for k, v in probabilities.items()},
            )

            return {
                "emotion":       mapped,
                "raw_emotion":   top_raw,
                "confidence":    confidence,
                "probabilities": probabilities,
                "gated":         False,
                "available":     True,
            }

        except Exception as exc:
            logger.exception("[AudioEmotion] analyse() failed: %s", exc)
            return _unavailable("inference_error")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _unavailable(reason: str = "unavailable") -> Dict[str, Any]:
    return {
        "emotion":       "neutral",
        "raw_emotion":   reason,
        "confidence":    0.0,
        "probabilities": {},
        "gated":         False,
        "available":     False,
    }
