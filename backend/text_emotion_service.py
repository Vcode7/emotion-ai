"""
text_emotion_service.py — Text Emotion Recognition via RoBERTa GoEmotions
=========================================================================

Uses the pretrained ``SamLowe/roberta-base-go_emotions`` model to classify
the emotional content of user text into 28 GoEmotions categories, then maps
the top label to a standard emotion vocabulary shared by the pipeline.

Model details:
    Checkpoint : SamLowe/roberta-base-go_emotions
    Task       : Multi-label text classification (GoEmotions 28-class)
    Input      : Plain English text
    Output     : Top emotion label + confidence + full probability dict
    Framework  : HuggingFace transformers (already a dependency)
    RAM        : ~500 MB for roberta-base

Lazy loading: the model is not loaded until the first call to ``analyse()``,
so startup time is unaffected if the service is never used.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GoEmotions → standard pipeline vocabulary mapping
# ---------------------------------------------------------------------------

# 28 GoEmotions labels, ordered as the model outputs them
_GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

# Map GoEmotions labels → standard vocabulary used by avatar / LLM context
_EMOTION_MAP: Dict[str, str] = {
    "admiration":    "happy",
    "amusement":     "happy",
    "anger":         "angry",
    "annoyance":     "angry",
    "approval":      "happy",
    "caring":        "empathetic",
    "confusion":     "thinking",
    "curiosity":     "thinking",
    "desire":        "happy",
    "disappointment":"sad",
    "disapproval":   "concerned",
    "disgust":       "angry",
    "embarrassment": "sad",
    "excitement":    "happy",
    "fear":          "fearful",
    "gratitude":     "happy",
    "grief":         "sad",
    "joy":           "happy",
    "love":          "happy",
    "nervousness":   "concerned",
    "optimism":      "happy",
    "pride":         "happy",
    "realization":   "thinking",
    "relief":        "neutral",
    "remorse":       "sad",
    "sadness":       "sad",
    "surprise":      "surprised",
    "neutral":       "neutral",
}


class TextEmotionService:
    """
    Singleton-safe service for text emotion classification.

    Attributes:
        available (bool): True when the model loaded successfully.
    """

    MODEL_NAME = "SamLowe/roberta-base-go_emotions"

    def __init__(self) -> None:
        self._pipeline = None
        self.available = False
        self._load()

    def _load(self) -> None:
        """Attempt to load the RoBERTa GoEmotions pipeline."""
        try:
            from transformers import pipeline as hf_pipeline  # noqa: PLC0415
            logger.info("Loading TextEmotionService: %s …", self.MODEL_NAME)
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self.MODEL_NAME,
                top_k=None,           # return all label scores
                truncation=True,
                max_length=512,
            )
            self.available = True
            logger.info("TextEmotionService ready.")
        except Exception as exc:
            logger.warning(
                "TextEmotionService could not load model (%s): %s — text emotion will be unavailable.",
                self.MODEL_NAME,
                exc,
            )
            self.available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, text: str) -> Dict[str, Any]:
        """
        Classify the emotional content of *text*.

        Returns
        -------
        dict with keys:
            emotion      (str)   — mapped standard emotion label
            raw_emotion  (str)   — original GoEmotions label
            confidence   (float) — probability of the top label [0.0, 1.0]
            probabilities (dict) — {raw_label: score} for all 28 classes
        """
        if not self.available or not self._pipeline:
            return _unavailable()

        if not text or not text.strip():
            return _unavailable()

        try:
            results = self._pipeline(text[:512])
            # With top_k=None, a single string returns [[{"label", "score"}, …]] on
            # most transformers versions but a flat [{"label", "score"}, …] on others.
            if results and isinstance(results[0], dict):
                scores: list = list(results)
            else:
                scores = list(results[0]) if results else []

            if not scores:
                return _unavailable()

            # Sort by score descending → pick top
            scores.sort(key=lambda x: x["score"], reverse=True)
            top = scores[0]
            raw_label: str = top["label"]
            confidence: float = round(float(top["score"]), 4)

            mapped = _EMOTION_MAP.get(raw_label, "neutral")

            probabilities = {
                item["label"]: round(float(item["score"]), 4)
                for item in scores
            }

            logger.debug(
                "[TextEmotion] text=%r → raw=%s (%.2f) → mapped=%s",
                text[:60],
                raw_label,
                confidence,
                mapped,
            )

            return {
                "emotion":       mapped,
                "raw_emotion":   raw_label,
                "confidence":    confidence,
                "probabilities": probabilities,
                "available":     True,
            }

        except Exception as exc:
            logger.exception("TextEmotionService.analyse failed: %s", exc)
            return _unavailable()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unavailable(reason: str = "unavailable") -> Dict[str, Any]:
    return {
        "emotion":       "neutral",
        "raw_emotion":   reason,
        "confidence":    0.0,
        "probabilities": {},
        "available":     False,
    }
