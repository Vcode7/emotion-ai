"""
routes/health.py — Health-check endpoints
==========================================

Provides two GET endpoints:

    GET /health         — Basic liveness check (quick).
    GET /health/models  — Detailed per-model status (useful for debugging).
"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(request: Request) -> dict:
    """
    Return the current health status of the backend.

    Response body:
        - ``status``        — always ``"ok"`` if this endpoint responds.
        - ``model_loaded``  — whether the DMESR model loaded successfully.
        - ``device``        — PyTorch compute device string.
        - ``sessions``      — number of active inference sessions.
        - ``tts_available`` — whether at least one TTS backend is active.
    """
    inference_manager = getattr(request.app.state, "inference_manager", None)
    tts_service = getattr(request.app.state, "tts_service", None)
    conversation_store = getattr(request.app.state, "conversation_store", None)

    return {
        "status": "ok",
        "model_loaded": inference_manager.model_loaded if inference_manager else False,
        "device": inference_manager.device if inference_manager else "unknown",
        "sessions": inference_manager.session_count if inference_manager else 0,
        "conversation_sessions": len(conversation_store) if conversation_store is not None else 0,
        "tts_available": tts_service.available if tts_service else False,
    }


@router.get("/health/models")
async def health_models(request: Request) -> dict:
    """
    Return detailed availability and metadata for every AI model/service.

    Use this to verify which models loaded at startup and to debug issues.

    Response body example::

        {
          "dsmr":         { "available": true,  "device": "cpu" },
          "whisper":      { "available": true,  "model": "openai/whisper-base" },
          "groq_whisper": { "available": true,  "model": "whisper-large-v3-turbo" },
          "text_emotion": { "available": true,  "model": "SamLowe/roberta-base-go_emotions" },
          "audio_emotion":{ "available": true,  "model": "superb/wav2vec2-base-superb-er",
                            "confidence_gate": 0.35, "labels": ["neu","hap","ang","sad"] },
          "face_emotion": { "available": false, "model": "HardlyHumans/Facial-expression-detection" },
          "tts":          { "available": true,  "backend": "edge-tts" }
        }
    """
    s = request.app.state

    inference_manager   = getattr(s, "inference_manager",    None)
    whisper_svc         = getattr(s, "whisper_service",       None)
    groq_svc            = getattr(s, "groq_whisper_service",  None)
    text_emo_svc        = getattr(s, "text_emotion_service",  None)
    audio_emo_svc       = getattr(s, "audio_emotion_service", None)
    face_emo_svc        = getattr(s, "face_emotion_service",  None)
    tts_svc             = getattr(s, "tts_service",           None)

    # ── Audio emotion labels (read from loaded model config) ──────────
    audio_labels: list = []
    if audio_emo_svc and audio_emo_svc.available and audio_emo_svc._id2label:
        audio_labels = list(audio_emo_svc._id2label.values())

    return {
        "dsmr": {
            "available": bool(inference_manager),
            "model_loaded": getattr(inference_manager, "model_loaded", False),
            "device": getattr(inference_manager, "device", "unknown"),
        },
        "whisper": {
            "available": getattr(whisper_svc, "available", False),
            "model": getattr(whisper_svc, "model_name", "openai/whisper-base"),
        },
        "groq_whisper": {
            "available": getattr(groq_svc, "available", False),
            "model": "whisper-large-v3-turbo",
            "api_key_set": bool(getattr(groq_svc, "api_key", None)),
        },
        "text_emotion": {
            "available": getattr(text_emo_svc, "available", False),
            "model": getattr(text_emo_svc, "MODEL_NAME", "SamLowe/roberta-base-go_emotions"),
        },
        "audio_emotion": {
            "available": getattr(audio_emo_svc, "available", False),
            "model": getattr(audio_emo_svc, "model_name", "superb/wav2vec2-base-superb-er"),
            "confidence_gate": 0.35,
            "labels": audio_labels,
        },
        "face_emotion": {
            "available": getattr(face_emo_svc, "available", False),
            "model": getattr(face_emo_svc, "model_name", "HardlyHumans/Facial-expression-detection"),
        },
        "tts": {
            "available": getattr(tts_svc, "available", False),
            "backend": getattr(tts_svc, "backend", "unknown"),
        },
    }
