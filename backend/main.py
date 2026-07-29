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
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# ─────────────────────────────────────────────────────────────────────────────
#  Route registration (no /api prefix — frontend proxy rewrites /api → /)
# ─────────────────────────────────────────────────────────────────────────────

from routes.health import router as health_router  # noqa: E402
from routes.chat import router as chat_router      # noqa: E402
from routes.tts import router as tts_router        # noqa: E402

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(tts_router)
