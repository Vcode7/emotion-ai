"""
routes/face.py — Standalone face emotion detection endpoint for testing & verification
========================================================================================

Allows uploading an image file (multipart) or base64-encoded image string to run
inference directly against the loaded FaceEmotionService (HardlyHumans/Facial-expression-detection).
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/face", tags=["face"])


class FaceDetectBase64Request(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded JPEG/PNG image data")


@router.post("/detect")
async def detect_face_emotion(
    request: Request,
    image: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """
    Classify facial emotion from an uploaded image (multipart form-data).
    """
    face_emotion_svc = getattr(request.app.state, "face_emotion_service", None)
    if not face_emotion_svc or not face_emotion_svc.available:
        raise HTTPException(
            status_code=503,
            detail="FaceEmotionService is not available or model failed to load",
        )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="No image file provided in multipart form-data (expected field 'image')",
        )

    t0 = time.perf_counter()
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty")

    result = face_emotion_svc.analyse_bytes(image_bytes)
    duration_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "success": True,
        "emotion": result.get("emotion", "neutral"),
        "raw_emotion": result.get("raw_emotion", "neutral"),
        "confidence": result.get("confidence", 0.0),
        "probabilities": result.get("probabilities", {}),
        "face_detected": result.get("face_detected", False),
        "available": result.get("available", True),
        "model": getattr(face_emotion_svc, "model_name", "HardlyHumans/Facial-expression-detection"),
        "inference_ms": duration_ms,
    }


@router.post("/detect-base64")
async def detect_face_emotion_base64(
    request: Request,
    body: FaceDetectBase64Request,
) -> Dict[str, Any]:
    """
    Classify facial emotion from a base64 encoded image string.
    """
    face_emotion_svc = getattr(request.app.state, "face_emotion_service", None)
    if not face_emotion_svc or not face_emotion_svc.available:
        raise HTTPException(
            status_code=503,
            detail="FaceEmotionService is not available or model failed to load",
        )

    t0 = time.perf_counter()
    raw_str = body.image_base64
    if "," in raw_str:
        raw_str = raw_str.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw_str)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {exc}")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Decoded image data is empty")

    result = face_emotion_svc.analyse_bytes(image_bytes)
    duration_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "success": True,
        "emotion": result.get("emotion", "neutral"),
        "raw_emotion": result.get("raw_emotion", "neutral"),
        "confidence": result.get("confidence", 0.0),
        "probabilities": result.get("probabilities", {}),
        "face_detected": result.get("face_detected", False),
        "available": result.get("available", True),
        "model": getattr(face_emotion_svc, "model_name", "HardlyHumans/Facial-expression-detection"),
        "inference_ms": duration_ms,
    }
