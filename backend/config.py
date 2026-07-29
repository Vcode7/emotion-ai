"""
config.py — Central configuration constants for the DMESR backend.
==================================================================

All path, device, and processing constants live here so that every other
module imports from a single source of truth.
"""

import os
from pathlib import Path

import torch

# ─────────────────────────────────────────────────────────────────────────────
#  Compute device
# ─────────────────────────────────────────────────────────────────────────────

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
"""PyTorch device string — 'cuda' when a GPU is available, else 'cpu'."""

# ─────────────────────────────────────────────────────────────────────────────
#  Directory layout (relative to this file's parent)
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_DIR: Path = Path(__file__).resolve().parent
"""Absolute path to the backend/ directory."""

PROJECT_ROOT: Path = BACKEND_DIR.parent
"""Absolute path to the project root (emotion-ai/)."""

# ── Load environment variables from .env ─────────────────────────────────────
def _load_dotenv() -> None:
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        try:
            with open(dotenv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                            v = v[1:-1]
                        os.environ[k] = v
                    elif line.startswith("gsk_"):
                        os.environ["GROQ_API_KEY"] = line
        except Exception as e:
            print(f"Error loading .env file: {e}")

_load_dotenv()

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

UPLOAD_DIR: Path = BACKEND_DIR / "uploads"
"""Root directory for temporary user uploads (audio, video)."""

UPLOAD_AUDIO_DIR: Path = UPLOAD_DIR / "audio"
"""Temporary storage for uploaded audio files."""

UPLOAD_VIDEO_DIR: Path = UPLOAD_DIR / "video"
"""Temporary storage for uploaded video files."""

GENERATED_DIR: Path = BACKEND_DIR / "generated"
"""Directory for generated output files (TTS audio, etc.)."""

# ─────────────────────────────────────────────────────────────────────────────
#  Whisper ASR settings
# ─────────────────────────────────────────────────────────────────────────────

WHISPER_MODEL_SIZE: str = "base"
"""Whisper model variant to load — 'tiny', 'base', 'small', 'medium', 'large'."""

# ─────────────────────────────────────────────────────────────────────────────
#  Audio processing
# ─────────────────────────────────────────────────────────────────────────────

MAX_AUDIO_DURATION: int = 30
"""Maximum audio duration in seconds accepted by upload endpoints."""

AUDIO_SAMPLE_RATE: int = 16000
"""Expected sample rate for all audio processing (Whisper, HuBERT)."""

# ─────────────────────────────────────────────────────────────────────────────
#  Video processing
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_NUM_FRAMES: int = 8
"""Number of frames uniformly sampled from each video clip."""

VIDEO_FRAME_SIZE: int = 112
"""Height and width (square) that each video frame is resized to."""

# ─────────────────────────────────────────────────────────────────────────────
#  Misc
# ─────────────────────────────────────────────────────────────────────────────

MAX_TEXT_LENGTH: int = 1024
"""Maximum character length for a single user text message."""

TTS_FILE_MAX_AGE: int = 300
"""TTS output files older than this many seconds are eligible for cleanup."""
