"""
face_emotion_service.py — Face Emotion Recognition via HardlyHumans/Facial-expression-detection
================================================================================================

Uses the pretrained Vision Transformer (ViT) model ``HardlyHumans/Facial-expression-detection``
to classify facial expressions from a cropped or full-frame image.

Model details:
    Checkpoint : HardlyHumans/Facial-expression-detection
    Architecture: ViTForImageClassification (google/vit-base-patch16-224 backbone)
    Input shape : (batch, 3, 224, 224) — RGB image processed by ViTImageProcessor
    Output      : 8 emotion classes (logits)
    Labels      : anger, contempt, disgust, fear, happy, neutral, sad, surprise
    Face detect : OpenCV Haar Cascade (haarcascade_frontalface_default.xml)
    Framework   : HuggingFace transformers + PyTorch

Label mapping → standard pipeline vocabulary:
    neutral   → neutral
    happy     → happy
    surprise  → surprised
    sad       → sad
    anger     → angry
    disgust   → angry
    fear      → fearful
    contempt  → angry
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAME = "HardlyHumans/Facial-expression-detection"

# Below this top-1 confidence the read is too uncertain → report neutral
# (mirrors the audio service's CONFIDENCE_GATE). The model reads many resting
# faces as "sad" at ~0.5 confidence.
FACE_CONFIDENCE_GATE = 0.45

_HAAR_SCALE = 1.1
_HAAR_MIN_NEIGHBORS = 5
_HAAR_MIN_SIZE = (30, 30)

_TO_STANDARD: Dict[str, str] = {
    "neutral":   "neutral",
    "happy":     "happy",
    "happiness": "happy",
    "surprise":  "surprised",
    "sad":       "sad",
    "sadness":   "sad",
    "anger":     "angry",
    "disgust":   "angry",
    "fear":      "fearful",
    "contempt":  "neutral",  # Subtle closed-lip smile / smirk; mapped to neutral instead of angry
}


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class FaceEmotionService:
    """
    Face emotion classifier using HardlyHumans/Facial-expression-detection ViT
    with OpenCV Haar face detection.

    Attributes:
        available (bool): True when model and Haar cascade are loaded.
        model_name (str): The Hugging Face checkpoint name.
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_name = _MODEL_NAME
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = None
        self._model = None
        self._face_cascade = None
        self._id2label: Dict[int, str] = {}
        self.available = False
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load Vision Transformer model and Haar cascade."""
        try:
            from transformers import ViTForImageClassification, ViTImageProcessor  # noqa: PLC0415

            logger.info("=" * 60)
            logger.info("[FaceEmotion] Loading model: %s", self.model_name)
            logger.info("[FaceEmotion] Device: %s", self._device)

            # 1. Image processor
            try:
                self._processor = ViTImageProcessor.from_pretrained(self.model_name)
            except Exception as proc_exc:
                logger.warning(
                    "[FaceEmotion] Direct processor load notice: %s. Falling back to base ViT processor.",
                    proc_exc,
                )
                from transformers import AutoImageProcessor  # noqa: PLC0415
                self._processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

            # 2. ViT model
            self._model = ViTForImageClassification.from_pretrained(self.model_name)
            self._model.to(self._device)
            self._model.eval()

            # Cache id2label mapping
            if hasattr(self._model.config, "id2label") and self._model.config.id2label:
                self._id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
            else:
                self._id2label = {
                    0: "anger", 1: "contempt", 2: "disgust", 3: "fear",
                    4: "happy", 5: "neutral",  6: "sad",     7: "surprise",
                }

            logger.info("[FaceEmotion] Model labels: %s", list(self._id2label.values()))

            # 3. Haar cascade for face detection
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(haar_path)
            if self._face_cascade.empty():
                logger.warning("[FaceEmotion] Haar cascade failed to load — face detection unavailable.")
                self._face_cascade = None

            self.available = True
            logger.info("[FaceEmotion] FaceEmotionService ready on %s.", self._device)
            logger.info("=" * 60)

        except Exception as exc:
            logger.warning("[FaceEmotion] FaceEmotionService could not load: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Detect faces and classify emotion from raw image bytes (JPEG/PNG/WebM frame).

        Parameters
        ----------
        image_bytes : bytes
            Raw image data (JPEG, PNG, or any OpenCV-decodable format).

        Returns
        -------
        dict with keys:
            emotion      (str)   — mapped standard emotion label
            raw_emotion  (str)   — model raw label
            confidence   (float) — softmax probability of top emotion
            probabilities (dict) — {label: score}
            face_detected (bool) — whether a face was found
            available    (bool)  — True
        """
        if not self.available or self._model is None or self._processor is None:
            return _unavailable()

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return _unavailable("decode_failed")

            return self._analyse_frame(img_bgr)

        except Exception as exc:
            logger.exception("FaceEmotionService.analyse_bytes failed: %s", exc)
            return _unavailable()

    def analyse_file(self, image_path: str) -> Dict[str, Any]:
        """Analyse emotion from an image file on disk."""
        if not self.available:
            return _unavailable()
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                return _unavailable("file_read_failed")
            return self._analyse_frame(img_bgr)
        except Exception as exc:
            logger.exception("FaceEmotionService.analyse_file failed: %s", exc)
            return _unavailable()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _analyse_frame(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Run face detection and ViT emotion inference on an OpenCV BGR frame."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        face_roi = None
        face_detected = False

        # ── Face detection ─────────────────────────────────────────────
        if self._face_cascade is not None and not self._face_cascade.empty():
            faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=_HAAR_SCALE,
                minNeighbors=_HAAR_MIN_NEIGHBORS,
                minSize=_HAAR_MIN_SIZE,
            )
            if len(faces) > 0:
                # Use the largest detected face with 18% contextual margin so smile & forehead aren't clipped
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]
                h_img, w_img = img_bgr.shape[:2]
                pad_x = int(w * 0.18)
                pad_y = int(h * 0.18)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w_img, x + w + pad_x)
                y2 = min(h_img, y + h + pad_y)
                face_roi = img_bgr[y1:y2, x1:x2]
                face_detected = True

        # Fallback: use full frame if no face detected
        if face_roi is None:
            face_roi = img_bgr

        # ── Preprocessing (BGR → RGB for ViT) ──────────────────────────
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=face_rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # ── ViT Inference ──────────────────────────────────────────────
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # Handle batch dim edge case
        if probs.ndim == 0:
            probs = np.array([float(probs)])

        probabilities = {
            self._id2label.get(i, f"class_{i}"): float(p)
            for i, p in enumerate(probs)
        }
        return _result_from_probabilities(probabilities, face_detected)

    def analyse_many(self, images: List[bytes]) -> Dict[str, Any]:
        """
        Analyse several frames of the same utterance and average them.

        Probabilities are averaged over the frames where a face was found,
        which is far steadier than a single snapshot taken after the user
        has stopped talking.
        """
        if not self.available:
            return _unavailable()
        results = [self.analyse_bytes(b) for b in images if b]
        if not results:
            return _unavailable("no_image")
        with_face = [r for r in results if r.get("available") and r.get("face_detected")]
        if not with_face:
            result = dict(results[-1])
        else:
            labels = {lbl for r in with_face for lbl in r.get("probabilities", {})}
            averaged = {
                lbl: sum(r["probabilities"].get(lbl, 0.0) for r in with_face) / len(with_face)
                for lbl in labels
            }
            result = _result_from_probabilities(averaged, face_detected=True)
        result["frames"] = len(results)
        result["frames_with_face"] = len(with_face)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_from_probabilities(probabilities: Dict[str, float], face_detected: bool) -> Dict[str, Any]:
    """Pick the winning label (ignoring 'contempt'), map it and apply the confidence gate."""
    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    # Ignore 'contempt' when selecting the winning emotion; pick next highest score
    non_contempt = [kv for kv in ranked if kv[0].lower() != "contempt"]
    top_raw_label, top_p = (non_contempt or ranked or [("neutral", 0.0)])[0]

    confidence = round(float(top_p), 4)
    mapped = _TO_STANDARD.get(top_raw_label, "neutral")
    gated = confidence < FACE_CONFIDENCE_GATE
    if gated:
        mapped = "neutral"

    logger.debug(
        "[FaceEmotion] face_detected=%s raw=%s (%.2f) → mapped=%s%s",
        face_detected, top_raw_label, confidence, mapped, " (gated)" if gated else "",
    )

    return {
        "emotion":       mapped,
        "raw_emotion":   top_raw_label,
        "confidence":    confidence,
        "probabilities": {k: round(float(v), 4) for k, v in probabilities.items()},
        "face_detected": face_detected,
        "gated":         gated,
        "available":     True,
    }


def _unavailable(reason: str = "unavailable") -> Dict[str, Any]:
    return {
        "emotion":       "neutral",
        "raw_emotion":   reason,
        "confidence":    0.0,
        "probabilities": {},
        "face_detected": False,
        "available":     False,
    }
