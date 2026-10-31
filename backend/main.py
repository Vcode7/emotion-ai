"""
main.py — FastAPI application entry-point for the DMESR backend
================================================================

Responsibilities:
    • Lifespan handler: initialise DMESR model, Whisper, TTS on startup.
    • CORS middleware for the Vite dev server (localhost:5173).
    • Static file mount for generated TTS audio.
    • Route registration (no /api prefix — the frontend proxy rewrites).

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import GENERATED_DIR, UPLOAD_AUDIO_DIR, UPLOAD_VIDEO_DIR, UPLOAD_DIR

# ─────────────────────────────────────────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dmesr.backend")


# ─────────────────────────────────────────────────────────────────────────────
#  Lifespan — startup & shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise heavyweight services once at startup and tear them down
    on shutdown.

    Services are stored on ``app.state`` so route handlers can access
    them via ``request.app.state.<service>``.
    """
    # ── Create directories ───────────────────────────────────────────────
    for d in (UPLOAD_AUDIO_DIR, UPLOAD_VIDEO_DIR, GENERATED_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Upload and generated directories ready.")

    # ── Load DMESR model ─────────────────────────────────────────────────
    logger.info("Loading DMESR inference manager …")
    from inference import InferenceManager

    app.state.inference_manager = InferenceManager()
    logger.info(
        "DMESR model loaded on %s.", app.state.inference_manager.device
    )

    # ── Load Whisper ─────────────────────────────────────────────────────
    logger.info("Loading Whisper ASR service …")
    from whisper_service import WhisperService

    app.state.whisper_service = WhisperService()
    if app.state.whisper_service.available:
        logger.info("Whisper ASR ready.")
    else:
        logger.warning("Whisper ASR unavailable — audio endpoints will return empty transcripts.")

    # ── Load TTS ─────────────────────────────────────────────────────────
    logger.info("Loading TTS service …")
    from tts_service import TTSService

    app.state.tts_service = TTSService()
    if app.state.tts_service.available:
        logger.info("TTS ready (backend: %s).", app.state.tts_service.backend)
    else:
        logger.warning("TTS unavailable — frontend should use Web Speech API.")

    # ── Load LLM service (stateless, no init needed) ─────────────────────
    from llm_service import LLMService

    app.state.llm_service = LLMService()
    logger.info("LLM service ready.")

    # ── Conversation memory (history, face baseline, emotion trajectory) ─
    from conversation_store import ConversationStore

    app.state.conversation_store = ConversationStore()
    logger.info("Conversation store ready.")

    # ── Load Groq Whisper service ─────────────────────────────────────
    logger.info("Loading Groq Whisper service …")
    from groq_whisper_service import GroqWhisperService

    app.state.groq_whisper_service = GroqWhisperService()
    if app.state.groq_whisper_service.available:
        logger.info("Groq Whisper service ready.")
    else:
        logger.warning("Groq Whisper service unavailable — no GROQ_API_KEY set.")

    # ── Load Text Emotion Service (RoBERTa GoEmotions) ───────────────
    logger.info("Loading Text Emotion Service (RoBERTa GoEmotions) …")
    from text_emotion_service import TextEmotionService

    app.state.text_emotion_service = TextEmotionService()
    if app.state.text_emotion_service.available:
        logger.info("Text Emotion Service ready.")
    else:
        logger.warning("Text Emotion Service unavailable — text emotion will return neutral.")

    # ── Load Audio Emotion Service (Wav2Vec2 SER) ──────────────────
    logger.info("Loading Audio Emotion Service (Wav2Vec2 SER) …")
    from audio_emotion_service import AudioEmotionService

    app.state.audio_emotion_service = AudioEmotionService()
    if app.state.audio_emotion_service.available:
        logger.info("Audio Emotion Service ready.")
    else:
        logger.warning("Audio Emotion Service unavailable — audio emotion will return neutral.")

    # ── Load Face Emotion Service (HardlyHumans ViT) ───────────────
    logger.info("Loading Face Emotion Service (HardlyHumans/Facial-expression-detection) …")
    from face_emotion_service import FaceEmotionService

    app.state.face_emotion_service = FaceEmotionService()
    if app.state.face_emotion_service.available:
        logger.info("Face Emotion Service ready.")
    else:
        logger.warning("Face Emotion Service unavailable — face emotion will return neutral.")

    # ── Load Emotion Logger (research JSONL) ────────────────────────
    from emotion_logger import EmotionLogger

    app.state.emotion_logger = EmotionLogger()
    logger.info("Emotion Logger ready.")

    # ── Model status report ──────────────────────────────────────────
    # Printed once at startup so you can immediately verify what loaded.
    _aes = app.state.audio_emotion_service
    _tes = app.state.text_emotion_service
    _fes = app.state.face_emotion_service
    _whs = app.state.whisper_service
    _gws = app.state.groq_whisper_service
    _tts = app.state.tts_service
    _inf = app.state.inference_manager

    def _ok(svc, extra=""):
        return f"✓ READY{(' — ' + extra) if extra else ''}"

    def _fail(label):
        return f"✗ UNAVAILABLE — {label} will be disabled"

    report_lines = [
        "",
        "╔══════════════════════════════════════════════════════════════╗",
        "║              DMESR MODEL STATUS REPORT                      ║",
        "╠══════════════════════════════════════════════════════════════╣",
        f"║  DSMR Inference       {_ok(_inf, f'device={_inf.device}'):<40}║",
        f"║  Local Whisper ASR    {'  ' + _ok(_whs) if _whs.available else '  ' + _fail('audio endpoints'):<40}║",
        f"║  Groq Whisper STT     {'  ' + _ok(_gws, 'whisper-large-v3-turbo') if _gws.available else '  ' + _fail('no GROQ_API_KEY'):<40}║",
        f"║  TTS Service          {'  ' + _ok(_tts, getattr(_tts, 'backend', '?')) if _tts.available else '  ' + _fail('TTS disabled'):<40}║",
        f"║  Text Emotion         {'  ' + _ok(_tes, 'roberta-base-go_emotions') if _tes.available else '  ' + _fail('text emotion disabled'):<40}║",
        f"║  Audio Emotion        {'  ' + _ok(_aes, getattr(_aes, 'model_name', '?').split('/')[-1]) if _aes.available else '  ' + _fail('audio emotion disabled'):<40}║",
        f"║  Face Emotion         {'  ' + _ok(_fes, getattr(_fes, 'model_name', '?').split('/')[-1]) if _fes.available else '  ' + _fail('face emotion disabled'):<40}║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
    ]
    for line in report_lines:
        logger.info(line)

    logger.info("═══ DMESR Backend startup complete ═══")

    yield  # ── application runs here ─────────────────────────────────────

    # ── Shutdown cleanup ─────────────────────────────────────────────────
    logger.info("DMESR Backend shutting down …")


# ─────────────────────────────────────────────────────────────────────────────
#  Application factory
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DMESR Emotion-AI Backend",
    description=(
        "Research-demo backend for the Dynamic Multimodal Emotion-State "
        "Representation (DMESR) model.  Provides multimodal chat endpoints "
        "with real-time emotion analysis, LLM response generation, and "
        "avatar / voice control parameters."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ─────────────────────────────────────────────────────────────────────────────
#  CORS — allow the Vite dev server
# ─────────────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:3000",   # alternate dev port
    ],
    allow_origin_regex=r"https?://.*\.ngrok(-free|\.free)?\.(dev|app|io)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
#  Static files — serve generated TTS audio
# ─────────────────────────────────────────────────────────────────────────────

GENERATED_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")
app.mount("/uploads",   StaticFiles(directory=str(UPLOAD_DIR)),    name="uploads")

# Mount avatar video output (served as /static/avatar/<filename>)
from pathlib import Path as _Path
_AVATAR_STATIC_DIR = _Path(__file__).resolve().parent / "static" / "avatar"
_AVATAR_STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/static/avatar",
    StaticFiles(directory=str(_AVATAR_STATIC_DIR)),
    name="avatar_static",
)

# Serve the static portrait image for frontend fallback
_ASSETS_DIR = _Path(__file__).resolve().parent / "assets"
_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/static/assets",
    StaticFiles(directory=str(_ASSETS_DIR)),
    name="assets_static",
)


# ─────────────────────────────────────────────────────────────────────────────
#  Route registration (no /api prefix — frontend proxy rewrites /api → /)
# ─────────────────────────────────────────────────────────────────────────────

from routes.health import router as health_router      # noqa: E402
from routes.chat import router as chat_router          # noqa: E402
from routes.tts import router as tts_router            # noqa: E402
from routes.transcribe import router as transcribe_router  # noqa: E402
from routes.face import router as face_router          # noqa: E402

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(tts_router)
app.include_router(transcribe_router)
app.include_router(face_router)
