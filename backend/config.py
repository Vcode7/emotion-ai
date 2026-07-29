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
