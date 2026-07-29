"""
routes/chat.py — Core chat endpoints (4 modality combinations)
================================================================

Each endpoint:
    1. Receives user input (text / audio / video, in various combos).
    2. Runs the DMESR inference pipeline to compute emotion state.
    3. Calls the LLM service with emotion context.
    4. Returns a unified response envelope.

Endpoints:
    POST /chat/text         — text-only
    POST /chat/audio        — audio upload (Whisper transcribes)
    POST /chat/text-video   — text + video upload
    POST /chat/audio-video  — audio + video upload

All endpoints return the same response schema — see ``ChatResponse``.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from audio_loader import load_audio

from config import (
    AUDIO_SAMPLE_RATE,
    GENERATED_DIR,
    MAX_AUDIO_DURATION,
    UPLOAD_AUDIO_DIR,
    UPLOAD_VIDEO_DIR,
)
from video_processor import extract_aligned_frames, extract_frames, frames_to_tensor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# ─────────────────────────────────────────────────────────────────────────────
#  Shared response schema
# ─────────────────────────────────────────────────────────────────────────────

class ResponseSegment(BaseModel):
    """A single LLM response segment with its emotion label."""
    text: str
    emotion: str


class ChatResponse(BaseModel):
    """Unified response envelope for all /chat/* endpoints."""
    transcript: str = Field("", description="User text (typed or transcribed).")
    emotion_state: dict = Field(default_factory=dict, description="11-D emotion state dict.")
    emotion_embedding: list = Field(default_factory=list, description="512-D emotion embedding.")
    emotion_timeline: list = Field(
        default_factory=list,
        description="Timeline of emotion states from memory.",
    )
    response: List[ResponseSegment] = Field(
        default_factory=list,
        description="LLM response segments.",
    )
    voice_params: dict = Field(default_factory=dict, description="TTS voice parameters.")
    avatar_commands: dict = Field(default_factory=dict, description="Avatar control dict.")
    audio_url: Optional[str] = Field(None, description="URL to the user's uploaded audio file.")


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_emotion_timeline(llm_context: dict) -> list:
    """
    Extract a flat list of ``{sentence, emotion_state}`` entries from the
    LLM context's ``recent_history`` for the frontend timeline visualisation.
    """
    history = llm_context.get("recent_history", [])
    return [
        {
            "sentence": entry.get("sentence", ""),
            "emotion_state": entry.get("emotion_state", {}),
            "timestamp": entry.get("timestamp", ""),
        }
        for entry in history
    ]


async def _save_upload(upload: UploadFile, dest_dir: Path) -> Path:
    """
    Persist an uploaded file to *dest_dir* with a UUID-based name.

    Returns the absolute path to the saved file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(upload.filename or "file").suffix or ".bin"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = dest_dir / filename

    content = await upload.read()
    with open(filepath, "wb") as f:
        f.write(content)
    return filepath


def _load_audio_waveform(audio_path: Path) -> torch.Tensor:
    """
    Load an audio file, resample to 16 kHz mono, and return a 1-D tensor.

    Truncates to ``MAX_AUDIO_DURATION`` seconds.
    """
    waveform, sr = load_audio(str(audio_path))

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != AUDIO_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=AUDIO_SAMPLE_RATE)
        waveform = resampler(waveform)

    # Flatten to 1-D and truncate
    waveform = waveform.squeeze(0)
    max_samples = MAX_AUDIO_DURATION * AUDIO_SAMPLE_RATE
    if waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]

    return waveform


def _safe_cleanup(path: Path) -> None:
    """Remove a file, ignoring errors."""
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  1. POST /chat/text  — text-only
# ─────────────────────────────────────────────────────────────────────────────

class TextChatRequest(BaseModel):
    """Body for ``POST /chat/text``."""
    text: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1)
    llm_provider: str = Field(..., description="openai | groq | gemini")
    llm_api_key: Optional[str] = Field(None, description="Optional API key. Falls back to server .env if not provided.")
    llm_model: Optional[str] = Field(None, description="Optional model override.")


@router.post("/chat/text", response_model=ChatResponse)
async def chat_text(body: TextChatRequest, request: Request) -> ChatResponse:
    """
    Process a text-only user message.

    Audio and video channels are zero-filled.
    """
    inference_manager = request.app.state.inference_manager
    llm_service = request.app.state.llm_service

    # ── DMESR inference ──────────────────────────────────────────────────
    result = inference_manager.process_text_only(body.session_id, body.text)

    # ── LLM response ─────────────────────────────────────────────────────
    llm_segments = await llm_service.generate_response(
        user_text=body.text,
        emotion_state=result["emotion_state"],
        llm_context=result["llm_context"],
        provider=body.llm_provider,
        api_key=body.llm_api_key,
        model=body.llm_model,
    )

    return ChatResponse(
        transcript=body.text,
        emotion_state=result["emotion_state"],
        emotion_embedding=result["emotion_embedding"],
        emotion_timeline=_build_emotion_timeline(result["llm_context"]),
        response=[ResponseSegment(**seg) for seg in llm_segments],
        voice_params=result["voice_params"],
        avatar_commands=result["avatar_commands"],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  2. POST /chat/audio  — audio upload
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/chat/audio", response_model=ChatResponse)
async def chat_audio(
    request: Request,
    audio: UploadFile = File(..., description="Audio file (WAV/MP3/WebM)."),
    session_id: str = Form(...),
    llm_provider: str = Form(...),
    llm_api_key: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
) -> ChatResponse:
    """
    Process an audio-only user message.

    1. Save uploaded audio.
    2. Transcribe with Whisper.
    3. Load waveform for DMESR.
    4. Run DMESR inference (audio + text, zero video).
    5. Call LLM.
    """
    inference_manager = request.app.state.inference_manager
    whisper_service = request.app.state.whisper_service
    llm_service = request.app.state.llm_service

    # ── Save audio ───────────────────────────────────────────────────────
    audio_path = await _save_upload(audio, UPLOAD_AUDIO_DIR)
    success = False

    try:
        # ── Whisper transcription ────────────────────────────────────────
        transcript_result = whisper_service.transcribe(str(audio_path))
        text = transcript_result.get("text", "").strip()
        if not text:
            text = "(inaudible)"

        # ── Load waveform for DMESR audio encoder ────────────────────────
        audio_waveform = _load_audio_waveform(audio_path)

        # ── DMESR inference ──────────────────────────────────────────────
        result = inference_manager.process_with_audio(
            session_id=session_id,
            text=text,
            audio_waveform=audio_waveform,
        )

        # ── LLM response ────────────────────────────────────────────────
        llm_segments = await llm_service.generate_response(
            user_text=text,
            emotion_state=result["emotion_state"],
            llm_context=result["llm_context"],
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model,
        )

        success = True
        return ChatResponse(
            transcript=text,
            emotion_state=result["emotion_state"],
            emotion_embedding=result["emotion_embedding"],
            emotion_timeline=_build_emotion_timeline(result["llm_context"]),
            response=[ResponseSegment(**seg) for seg in llm_segments],
            voice_params=result["voice_params"],
            avatar_commands=result["avatar_commands"],
            audio_url=f"/uploads/audio/{audio_path.name}",
        )

    finally:
        if not success:
            _safe_cleanup(audio_path)


# ─────────────────────────────────────────────────────────────────────────────
#  3. POST /chat/text-video  — text + video upload
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/chat/text-video", response_model=ChatResponse)
async def chat_text_video(
    request: Request,
    video: UploadFile = File(..., description="Video file (MP4/WebM)."),
    text: str = Form(...),
    session_id: str = Form(...),
    llm_provider: str = Form(...),
    llm_api_key: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
) -> ChatResponse:
    """
    Process a text message with an accompanying video.

    Video frames are extracted and fed to the DMESR VideoEncoder.
    Audio channel is zero-filled.
    """
    inference_manager = request.app.state.inference_manager
    llm_service = request.app.state.llm_service

    # ── Save video ───────────────────────────────────────────────────────
    video_path = await _save_upload(video, UPLOAD_VIDEO_DIR)

    try:
        # ── Extract frames ───────────────────────────────────────────────
        raw_frames = extract_frames(str(video_path))      # (8, 112, 112, 3)
        video_tensor = frames_to_tensor(raw_frames)       # (3, 8, 112, 112)

        # ── DMESR inference ──────────────────────────────────────────────
        result = inference_manager.process_with_video(
            session_id=session_id,
            text=text,
            video_frames_tensor=video_tensor,
        )

        # ── LLM response ────────────────────────────────────────────────
        llm_segments = await llm_service.generate_response(
            user_text=text,
            emotion_state=result["emotion_state"],
            llm_context=result["llm_context"],
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model,
        )

        return ChatResponse(
            transcript=text,
            emotion_state=result["emotion_state"],
            emotion_embedding=result["emotion_embedding"],
            emotion_timeline=_build_emotion_timeline(result["llm_context"]),
            response=[ResponseSegment(**seg) for seg in llm_segments],
            voice_params=result["voice_params"],
            avatar_commands=result["avatar_commands"],
        )

    finally:
        _safe_cleanup(video_path)


# ─────────────────────────────────────────────────────────────────────────────
#  4. POST /chat/audio-video  — audio + video upload
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/chat/audio-video", response_model=ChatResponse)
async def chat_audio_video(
    request: Request,
    audio: UploadFile = File(..., description="Audio file (WAV/MP3/WebM)."),
    video: UploadFile = File(..., description="Video file (MP4/WebM)."),
    session_id: str = Form(...),
    llm_provider: str = Form(...),
    llm_api_key: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
) -> ChatResponse:
    """
    Process a full multimodal message: audio + video.

    1. Save both files.
    2. Transcribe audio with Whisper (segment-level timestamps).
    3. Extract video frames aligned to the speech segment.
    4. Run full DMESR inference (text + audio + video).
    5. Call LLM.
    """
    inference_manager = request.app.state.inference_manager
    whisper_service = request.app.state.whisper_service
    llm_service = request.app.state.llm_service

    # ── Save uploads ─────────────────────────────────────────────────────
    audio_path = await _save_upload(audio, UPLOAD_AUDIO_DIR)
    video_path = await _save_upload(video, UPLOAD_VIDEO_DIR)
    audio_success = False

    try:
        # ── Whisper transcription (with segments) ────────────────────────
        transcript_result = whisper_service.transcribe(str(audio_path))
        text = transcript_result.get("text", "").strip()
        segments = transcript_result.get("segments", [])

        if not text:
            text = "(inaudible)"

        # ── Determine time window for frame alignment ────────────────────
        if segments:
            start_time = segments[0].get("start", 0.0)
            end_time = segments[-1].get("end", 5.0)
        else:
            start_time = 0.0
            end_time = 5.0

        # ── Extract aligned video frames ─────────────────────────────────
        raw_frames = extract_aligned_frames(
            str(video_path),
            start_time=start_time,
            end_time=end_time,
        )
        video_tensor = frames_to_tensor(raw_frames)  # (3, 8, 112, 112)

        # ── Load audio waveform ──────────────────────────────────────────
        audio_waveform = _load_audio_waveform(audio_path)

        # ── DMESR inference (all modalities) ─────────────────────────────
        result = inference_manager.process_full(
            session_id=session_id,
            text=text,
            audio_waveform=audio_waveform,
            video_frames_tensor=video_tensor,
        )

        # ── LLM response ────────────────────────────────────────────────
        llm_segments = await llm_service.generate_response(
            user_text=text,
            emotion_state=result["emotion_state"],
            llm_context=result["llm_context"],
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model,
        )

        audio_success = True
        return ChatResponse(
            transcript=text,
            emotion_state=result["emotion_state"],
            emotion_embedding=result["emotion_embedding"],
            emotion_timeline=_build_emotion_timeline(result["llm_context"]),
            response=[ResponseSegment(**seg) for seg in llm_segments],
            voice_params=result["voice_params"],
            avatar_commands=result["avatar_commands"],
            audio_url=f"/uploads/audio/{audio_path.name}",
        )

    finally:
        if not audio_success:
            _safe_cleanup(audio_path)
        _safe_cleanup(video_path)
