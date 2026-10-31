"""
routes/chat.py — Core chat endpoints
=====================================

Each endpoint:
    1. Receives user input (text / audio / video / webcam frames).
    2. Gates out non-speech (Whisper hallucinations on silence / noise).
    3. Runs the emotion experts in parallel — text (RoBERTa GoEmotions),
       voice (Wav2Vec2 SER), face (ViT), and DMESR sentiment — and fuses
       them into one user-emotion read (``emotion_fusion.fuse``).
    4. Calls the LLM with the fused read, the emotional trajectory and the
       recent conversation history (``conversation_store``).
    5. Returns a response envelope.

Endpoints:
    POST /chat/text          — text-only
    POST /chat/audio         — audio upload (Whisper transcribes)
    POST /chat/text-video    — text + video upload
    POST /chat/audio-video   — audio + video upload
    POST /chat/visual        — Visual Chat: audio and/or text + webcam frames,
                               emotion(sentence) response format
    POST /chat/call-complete — log a finished Visual Chat call

The first four return ``ChatResponse``; /chat/visual returns ``VisualChatResponse``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torchaudio
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from audio_loader import load_audio
from config import (
    AUDIO_SAMPLE_RATE,
    MAX_AUDIO_DURATION,
    UPLOAD_AUDIO_DIR,
    UPLOAD_VIDEO_DIR,
)
from emotion_fusion import dsmr_sentiment, fuse, nonverbal_reads
from llm_service import response_voice_params
from speech_gate import SpeechStats, analyse_speech, is_no_speech, trim_silence
from tts_service import TTSService
from video_processor import extract_aligned_frames, extract_frames, frames_to_tensor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

UPLOAD_MAX_AGE_SECONDS = 3600
"""Uploaded audio is kept this long so the UI can replay it, then deleted."""

NO_SPEECH_REPLY = "I didn't quite catch that — could you say it again?"


# ─────────────────────────────────────────────────────────────────────────────
#  Shared response schema
# ─────────────────────────────────────────────────────────────────────────────

class ResponseSegment(BaseModel):
    """A single LLM response segment with its emotion label."""
    text: str
    emotion: str


class ChatResponse(BaseModel):
    """Unified response envelope for the /chat/text|audio|text-video|audio-video endpoints."""
    transcript: str = Field("", description="User text (typed or transcribed).")
    user_emotion: Optional[str] = Field(None, description="Fused user emotion label (e.g. Sad, Happy, Angry).")
    user_emotion_score: Optional[float] = Field(None, description="Confidence [0.0-1.0] of the fused user emotion.")
    emotion_state: dict = Field(default_factory=dict, description="DMESR emotion state dict (sentiment-derived).")
    emotion_embedding: list = Field(default_factory=list, description="512-D emotion embedding.")
    emotion_timeline: list = Field(
        default_factory=list,
        description="Timeline of emotion states from memory.",
    )
    response: List[ResponseSegment] = Field(
        default_factory=list,
        description="LLM response segments.",
    )
    voice_params: dict = Field(default_factory=dict, description="TTS voice parameters for the AI's reply.")
    avatar_commands: dict = Field(default_factory=dict, description="Avatar control dict.")
    audio_url: Optional[str] = Field(None, description="URL to the user's uploaded audio file.")
    fused_emotion: dict = Field(default_factory=dict, description="Full fused multimodal emotion read.")
    no_speech: bool = Field(False, description="True when the audio contained no usable speech.")


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers — I/O
# ─────────────────────────────────────────────────────────────────────────────

def _build_emotion_timeline(llm_context: dict) -> list:
    """
    Extract a flat list of ``{sentence, emotion_state}`` entries from the
    DMESR context's ``recent_history`` for the frontend timeline visualisation.
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
    Persist an uploaded file to *dest_dir* with a UUID-based name and prune
    uploads older than ``UPLOAD_MAX_AGE_SECONDS``.

    Returns the absolute path to the saved file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(upload.filename or "file").suffix or ".bin"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = dest_dir / filename

    content = await upload.read()
    with open(filepath, "wb") as f:
        f.write(content)

    try:
        TTSService.cleanup_old_files(dest_dir, max_age_seconds=UPLOAD_MAX_AGE_SECONDS)
    except Exception:  # non-critical
        logger.debug("Upload cleanup failed for %s", dest_dir, exc_info=True)
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


def _safe_cleanup(path: Optional[Path]) -> None:
    """Remove a file, ignoring errors."""
    if path is None:
        return
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


async def _transcribe(
    request: Request,
    audio_path: Path,
    stt_provider: Optional[str],
    groq_api_key: Optional[str],
) -> Dict[str, Any]:
    """Transcribe with Groq (if requested and reachable) or local Whisper."""
    whisper_service = request.app.state.whisper_service
    if stt_provider == "groq":
        groq_service = getattr(request.app.state, "groq_whisper_service", None)
        if groq_service is not None and (groq_service.available or groq_api_key):
            result = await groq_service.transcribe_file(
                str(audio_path), api_key_override=groq_api_key or None
            )
            if not result.get("error"):
                result["provider"] = "groq"
                return result
            logger.warning("Groq STT failed (%s) — falling back to local Whisper.", result["error"])
        else:
            logger.warning("Groq STT requested but unavailable — using local Whisper.")
    result = await asyncio.to_thread(whisper_service.transcribe, str(audio_path))
    result["provider"] = "whisper"
    return result


async def _load_speech(audio_path: Path) -> Tuple[torch.Tensor, torch.Tensor, SpeechStats]:
    """Load a waveform and return (full, silence-trimmed, speech stats)."""
    waveform = await asyncio.to_thread(_load_audio_waveform, audio_path)
    stats = analyse_speech(waveform, AUDIO_SAMPLE_RATE)
    return waveform, trim_silence(waveform, AUDIO_SAMPLE_RATE, stats=stats), stats


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers — emotion analysis + LLM turn
# ─────────────────────────────────────────────────────────────────────────────

def _unavailable(**extra: Any) -> Dict[str, Any]:
    return {"emotion": "neutral", "confidence": 0.0, "probabilities": {}, "available": False, **extra}


async def _analyse_turn(
    request: Request,
    session_id: str,
    text: str,
    speech_waveform: Optional[torch.Tensor] = None,
    images: Optional[List[bytes]] = None,
    video_tensor: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Run DMESR + text + voice + face experts in parallel (off the event loop)
    and fuse them, using and updating this session's emotion state.
    """
    state = request.app.state
    inference_manager = state.inference_manager
    text_svc = getattr(state, "text_emotion_service", None)
    audio_svc = getattr(state, "audio_emotion_service", None)
    face_svc = getattr(state, "face_emotion_service", None)
    latency: Dict[str, float] = {}

    async def timed(key: str, fn: Callable, *args: Any) -> Any:
        t0 = time.perf_counter()
        result = await asyncio.to_thread(fn, *args)
        latency[key] = _ms(t0)
        return result

    async def run_dsmr() -> Dict[str, Any]:
        if speech_waveform is not None and video_tensor is not None:
            return await timed("dsmr_ms", inference_manager.process_full,
                               session_id, text, speech_waveform, video_tensor)
        if speech_waveform is not None:
            return await timed("dsmr_ms", inference_manager.process_with_audio,
                               session_id, text, speech_waveform)
        if video_tensor is not None:
            return await timed("dsmr_ms", inference_manager.process_with_video,
                               session_id, text, video_tensor)
        return await timed("dsmr_ms", inference_manager.process_text_only, session_id, text)

    async def run_text() -> Dict[str, Any]:
        if text_svc and text_svc.available and text:
            return await timed("text_emotion_ms", text_svc.analyse, text)
        return _unavailable()

    async def run_audio() -> Dict[str, Any]:
        if audio_svc and audio_svc.available and speech_waveform is not None:
            return await timed("audio_emotion_ms", audio_svc.analyse, speech_waveform)
        return _unavailable()

    async def run_face() -> Dict[str, Any]:
        if face_svc and face_svc.available and images:
            return await timed("face_emotion_ms", face_svc.analyse_many, images)
        return _unavailable(face_detected=False)

    dsmr_result, text_res, audio_res, face_res = await asyncio.gather(
        run_dsmr(), run_text(), run_audio(), run_face(),
    )

    convo = state.conversation_store.get(session_id)
    dsmr = dsmr_sentiment(dsmr_result["emotion_state"])
    fused = fuse(
        text=text_res,
        audio=audio_res,
        face=face_res,
        dsmr_valence=dsmr["valence"],
        user_text=text,
        face_baseline=convo.face_baseline,
        face_observations=convo.face_observations,
    )
    # Baseline is updated after calibration so a frame never cancels itself out.
    convo.update_face_baseline(fused.pop("face_raw_distribution", None))
    if fused["confidence"] > 0:
        convo.update_fused(fused["distribution"])
    smoothed = convo.smoothed_emotion()
    if smoothed:
        fused["smoothed"] = {"emotion": smoothed[0], "confidence": smoothed[1]}

    logger.info(
        "[Emotion] session=%s text=%s audio=%s face=%s dsmr=%s(%+.2f) → fused=%s %.0f%% (%s)%s",
        session_id[:8], text_res.get("emotion"), audio_res.get("emotion"),
        face_res.get("emotion") if face_res.get("face_detected") else "-",
        dsmr["emotion"], dsmr["valence"], fused["emotion"], fused["confidence"] * 100,
        fused["agreement"], f" CONFLICT={fused['conflict']}" if fused.get("conflict") else "",
    )

    return {
        "dsmr_result": dsmr_result,
        "dsmr": dsmr,
        "fused": fused,
        "text": text_res,
        "audio": audio_res,
        "face": face_res,
        "latency": latency,
    }


def _is_error_reply(text: str) -> bool:
    return text.startswith(("LLM error", "Unknown LLM provider", "I'm sorry — I couldn't connect"))


async def _llm_turn(
    request: Request,
    session_id: str,
    text: str,
    analysis: Dict[str, Any],
    generate: Callable[..., Any],
    to_reply: Callable[[Any], Tuple[str, str]],
) -> Any:
    """
    Run one LLM turn with conversation memory.

    *generate* is an LLMService method; *to_reply* turns its result into
    ``(reply_text, reply_emotion)`` for the history.
    """
    convo = request.app.state.conversation_store.get(session_id)
    fused = analysis["fused"]

    history = convo.history_messages()          # before adding the current turn
    convo.add_user_turn(text, fused["emotion"], fused["confidence"], nonverbal_reads(fused))
    allow_conflict = convo.may_raise_conflict()

    result = await generate(
        user_text=text,
        fused=fused,
        dsmr=analysis["dsmr"],
        history=history,
        trajectory=convo.trajectory(),
        allow_conflict_mention=allow_conflict,
    )

    reply_text, reply_emotion = to_reply(result)
    if not _is_error_reply(reply_text):
        if fused.get("conflict") and allow_conflict:
            convo.mark_conflict_raised()
        convo.add_assistant_turn(reply_text, reply_emotion)
    return result


async def _respond_segments(
    request: Request,
    session_id: str,
    text: str,
    analysis: Dict[str, Any],
    provider: str,
    api_key: Optional[str],
    model: Optional[str],
) -> List[Dict[str, str]]:
    llm_service = request.app.state.llm_service

    async def generate(**kwargs: Any) -> List[Dict[str, str]]:
        return await llm_service.generate_response(
            provider=provider, api_key=api_key, model=model, **kwargs,
        )

    return await _llm_turn(
        request, session_id, text, analysis, generate,
        lambda segs: (
            " ".join(s.get("text", "") for s in segs),
            segs[0].get("emotion", "neutral") if segs else "neutral",
        ),
    )


def _user_emotion_label(analysis: Dict[str, Any]) -> Tuple[str, float]:
    """Fused label (capitalised for display); DMESR sentiment if no expert was usable."""
    fused = analysis["fused"]
    if fused["confidence"] > 0:
        return fused["emotion"].capitalize(), round(float(fused["confidence"]), 2)
    dsmr = analysis["dsmr"]
    return dsmr["emotion"].capitalize(), round(float(dsmr["confidence"]), 2)


def _chat_response(
    transcript: str,
    analysis: Dict[str, Any],
    segments: List[Dict[str, str]],
    audio_url: Optional[str] = None,
) -> ChatResponse:
    dsmr_result = analysis["dsmr_result"]
    user_emo, user_score = _user_emotion_label(analysis)
    reply_emotion = segments[0]["emotion"] if segments else "neutral"
    return ChatResponse(
        transcript=transcript,
        user_emotion=user_emo,
        user_emotion_score=user_score,
        emotion_state=dsmr_result["emotion_state"],
        emotion_embedding=dsmr_result["emotion_embedding"],
        emotion_timeline=_build_emotion_timeline(dsmr_result["llm_context"]),
        response=[ResponseSegment(**seg) for seg in segments],
        # The avatar and voice express the AI's reply, not the DMESR read of the user.
        voice_params=response_voice_params(reply_emotion),
        avatar_commands={**(dsmr_result.get("avatar_commands") or {}), "expression": reply_emotion},
        audio_url=audio_url,
        fused_emotion=analysis["fused"],
    )


def _no_speech_response(transcript: str, audio_url: Optional[str]) -> ChatResponse:
    return ChatResponse(
        transcript=transcript,
        response=[ResponseSegment(text=NO_SPEECH_REPLY, emotion="neutral")],
        voice_params=response_voice_params("neutral"),
        avatar_commands={"expression": "neutral"},
        audio_url=audio_url,
        no_speech=True,
    )


def _log_simple_turn(
    request: Request,
    session_id: str,
    text: str,
    analysis: Dict[str, Any],
    segments: List[Dict[str, str]],
    channel: str,
) -> None:
    emotion_logger = getattr(request.app.state, "emotion_logger", None)
    if not emotion_logger:
        return
    user_emo, user_score = _user_emotion_label(analysis)
    emotion_logger.log_simple_turn(
        session_id=session_id,
        user_text=text,
        user_emotion=user_emo,
        user_score=user_score,
        ai_response_text=" ".join(seg.get("text", "") for seg in segments),
        ai_response_emotion=segments[0].get("emotion", "neutral") if segments else "neutral",
        channel=channel,
    )


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
    """Process a text-only user message (audio and video channels are zero-filled for DMESR)."""
    text = body.text.strip()
    analysis = await _analyse_turn(request, body.session_id, text)
    segments = await _respond_segments(
        request, body.session_id, text, analysis,
        body.llm_provider, body.llm_api_key, body.llm_model,
    )
    _log_simple_turn(request, body.session_id, text, analysis, segments, "text")
    return _chat_response(text, analysis, segments)


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
    stt_provider: Optional[str] = Form(None, description="STT provider: 'whisper' (default) or 'groq'."),
    groq_api_key: Optional[str] = Form(None, description="Groq API key for Groq Whisper STT."),
) -> ChatResponse:
    """
    Process an audio-only user message: transcribe, gate non-speech,
    analyse text + voice + DMESR, then call the LLM.
    """
    audio_path = await _save_upload(audio, UPLOAD_AUDIO_DIR)
    audio_url = f"/uploads/audio/{audio_path.name}"
    success = False

    try:
        transcript = await _transcribe(request, audio_path, stt_provider, groq_api_key)
        text = transcript.get("text", "").strip()
        _, speech_waveform, stats = await _load_speech(audio_path)

        no_speech, reason = is_no_speech(text, transcript.get("segments"), stats.voiced_sec)
        if no_speech:
            logger.info("/chat/audio: no speech (%s) — transcript %r", reason, text)
            success = True
            return _no_speech_response(text or "(inaudible)", audio_url)

        analysis = await _analyse_turn(request, session_id, text, speech_waveform=speech_waveform)
        segments = await _respond_segments(
            request, session_id, text, analysis, llm_provider, llm_api_key, llm_model,
        )
        _log_simple_turn(request, session_id, text, analysis, segments, "audio")

        success = True
        return _chat_response(text, analysis, segments, audio_url)

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
    """Process a text message with an accompanying video (fed to DMESR)."""
    video_path = await _save_upload(video, UPLOAD_VIDEO_DIR)
    text = text.strip()

    try:
        raw_frames = await asyncio.to_thread(extract_frames, str(video_path))   # (8, 112, 112, 3)
        video_tensor = frames_to_tensor(raw_frames)                             # (3, 8, 112, 112)

        analysis = await _analyse_turn(request, session_id, text, video_tensor=video_tensor)
        segments = await _respond_segments(
            request, session_id, text, analysis, llm_provider, llm_api_key, llm_model,
        )
        _log_simple_turn(request, session_id, text, analysis, segments, "text-video")
        return _chat_response(text, analysis, segments)

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
    stt_provider: Optional[str] = Form(None, description="STT provider: 'whisper' (default) or 'groq'."),
    groq_api_key: Optional[str] = Form(None, description="Groq API key for Groq Whisper STT."),
) -> ChatResponse:
    """
    Process a full multimodal message: audio + video.

    Video frames are extracted around the transcribed speech segment.
    """
    audio_path = await _save_upload(audio, UPLOAD_AUDIO_DIR)
    video_path = await _save_upload(video, UPLOAD_VIDEO_DIR)
    audio_url = f"/uploads/audio/{audio_path.name}"
    audio_success = False

    try:
        transcript = await _transcribe(request, audio_path, stt_provider, groq_api_key)
        text = transcript.get("text", "").strip()
        segments_stt = transcript.get("segments", [])
        _, speech_waveform, stats = await _load_speech(audio_path)

        no_speech, reason = is_no_speech(text, segments_stt, stats.voiced_sec)
        if no_speech:
            logger.info("/chat/audio-video: no speech (%s) — transcript %r", reason, text)
            audio_success = True
            return _no_speech_response(text or "(inaudible)", audio_url)

        # ── Frame window: speech segments if known, else the voiced region ──
        if segments_stt:
            start_time = segments_stt[0].get("start", 0.0)
            end_time = segments_stt[-1].get("end", 5.0)
        else:
            start_time = stats.start_sec or 0.0
            end_time = stats.end_sec or 5.0

        raw_frames = await asyncio.to_thread(
            extract_aligned_frames, str(video_path), start_time, end_time,
        )
        video_tensor = frames_to_tensor(raw_frames)  # (3, 8, 112, 112)

        analysis = await _analyse_turn(
            request, session_id, text,
            speech_waveform=speech_waveform, video_tensor=video_tensor,
        )
        segments = await _respond_segments(
            request, session_id, text, analysis, llm_provider, llm_api_key, llm_model,
        )
        _log_simple_turn(request, session_id, text, analysis, segments, "audio-video")

        audio_success = True
        return _chat_response(text, analysis, segments, audio_url)

    finally:
        if not audio_success:
            _safe_cleanup(audio_path)
        _safe_cleanup(video_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Call Completion Endpoint
# ─────────────────────────────────────────────────────────────────────────────

class CallCompleteRequest(BaseModel):
    session_id: str
    turns: Optional[List[Dict[str, Any]]] = None
    duration_seconds: Optional[float] = None


@router.post("/chat/call-complete")
async def call_complete_endpoint(request: Request, body: CallCompleteRequest) -> Dict[str, Any]:
    """
    Log full call session summary when user ends a call.
    """
    emotion_logger = getattr(request.app.state, "emotion_logger", None)
    if emotion_logger:
        emotion_logger.log_call_session(
            session_id=body.session_id,
            turns=body.turns or [],
            duration_seconds=body.duration_seconds,
        )
    return {"status": "ok", "logged_turns": len(body.turns or [])}


# ─────────────────────────────────────────────────────────────────────────────
#  5. POST /chat/visual  — Visual Chat (fused multimodal emotion + LLM)
# ─────────────────────────────────────────────────────────────────────────────

class VisualChatResponse(BaseModel):
    """Response envelope for POST /chat/visual."""
    transcript:       str = Field("", description="Transcribed user speech.")

    # ── Emotion expert outputs ────────────────────────────────────────
    text_emotion:     Dict[str, Any] = Field(default_factory=dict)
    audio_emotion:    Dict[str, Any] = Field(default_factory=dict)
    face_emotion:     Dict[str, Any] = Field(default_factory=dict)
    dsmr_emotion:     Dict[str, Any] = Field(default_factory=dict, description="DMESR sentiment read.")
    fused_emotion:    Dict[str, Any] = Field(default_factory=dict, description="Fused user emotion read.")

    # ── LLM response in emotion(sentence) format ────────────────────────
    llm_emotion:      str = Field("neutral", description="AI response emotion.")
    llm_sentence:     str = Field("", description="AI response sentence.")

    # ── DMESR fields (for avatar / voice continuity) ────────────────────
    emotion_state:    Dict[str, Any] = Field(default_factory=dict)
    emotion_embedding: list = Field(default_factory=list)
    voice_params:     Dict[str, Any] = Field(default_factory=dict)
    avatar_commands:  Dict[str, Any] = Field(default_factory=dict)
    audio_url:        Optional[str]  = Field(None)

    # ── Speech gate ──────────────────────────────────────────────────────
    no_speech:        bool = Field(False, description="True when the utterance had no usable speech; nothing else is filled.")
    no_speech_reason: Optional[str] = Field(None)

    # ── Research logging ──────────────────────────────────────────
    turn_id:          Optional[str]  = Field(None, description="Research log turn ID.")
    latency:          Dict[str, Any] = Field(default_factory=dict)


@router.post("/chat/visual", response_model=VisualChatResponse)
async def chat_visual(
    request:      Request,
    audio:        Optional[UploadFile] = File(None, description="Audio file (WAV/MP3/WebM)."),
    user_text:    Optional[str] = Form(None, description="Optional pre-transcribed user text."),
    session_id:   str        = Form(...),
    llm_provider: str        = Form(...),
    llm_api_key:  Optional[str] = Form(None),
    llm_model:    Optional[str] = Form(None),
    stt_provider: Optional[str] = Form(None, description="STT provider: 'whisper' or 'groq'."),
    groq_api_key: Optional[str] = Form(None),
    image:        Optional[UploadFile] = File(None, description="Optional webcam frame (JPEG/PNG)."),
    # Not Optional[List[...]]: FastAPI then parses only the last file of a multi-file field.
    images:       List[UploadFile] = File(default=[], description="Optional webcam frames captured during speech."),
) -> VisualChatResponse:
    """
    Visual Chat endpoint.

    Pipeline:
        1. Save audio (if supplied) + read webcam frames.
        2. STT (Groq / local Whisper) or direct ``user_text``.
        3. Speech gate — hallucinated / silent utterances return ``no_speech``.
        4. In parallel: text emotion, voice SER, face (multi-frame), DMESR.
        5. Fuse into one user-emotion read (session-calibrated face baseline).
        6. LLM with fused read + trajectory + history → emotion(sentence).
        7. Log to JSONL research file.
    """
    t_total_start = time.perf_counter()
    latency: Dict[str, Optional[float]] = {
        k: None for k in ["stt_ms", "text_emotion_ms", "audio_emotion_ms",
                           "face_emotion_ms", "dsmr_ms", "llm_ms", "total_ms"]
    }
    llm_service    = request.app.state.llm_service
    emotion_logger = getattr(request.app.state, "emotion_logger", None)

    audio_path: Optional[Path] = None
    audio_success = False
    if audio is not None and getattr(audio, "filename", None):
        try:
            audio_path = await _save_upload(audio, UPLOAD_AUDIO_DIR)
        except Exception as e:
            logger.warning("/chat/visual: failed to save audio: %s", e)

    image_bytes: List[bytes] = []
    for upload in ([image] if image is not None else []) + list(images or []):
        try:
            data = await upload.read()
            if data:
                image_bytes.append(data)
        except Exception as e:
            logger.warning("/chat/visual: failed to read image: %s", e)

    try:
        # ── Step 1: STT or direct user_text ──────────────────────────
        typed = bool(user_text and user_text.strip())
        stt_segments: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        if typed:
            text = user_text.strip()
            latency["stt_ms"] = 0.0
            logger.info("[VisualChat] Using direct user_text: %r", text[:120])
        elif audio_path is not None:
            logger.info("[STT] Audio received -- %.1f KB | stt_provider=%s",
                        audio_path.stat().st_size / 1024, stt_provider or "not-set")
            transcript = await _transcribe(request, audio_path, stt_provider, groq_api_key)
            text = transcript.get("text", "").strip()
            stt_segments = transcript.get("segments") or []
            latency["stt_ms"] = _ms(t0)
            logger.info("[STT] Transcription complete -- %r (stt_ms=%.0f, via=%s)",
                        text[:120], latency["stt_ms"], transcript.get("provider"))
        else:
            text = ""
            latency["stt_ms"] = 0.0

        # ── Step 2: waveform + speech gate ───────────────────────────
        speech_waveform: Optional[torch.Tensor] = None
        stats: Optional[SpeechStats] = None
        if audio_path is not None:
            _, speech_waveform, stats = await _load_speech(audio_path)

        if not typed:
            no_speech, reason = is_no_speech(text, stt_segments, stats.voiced_sec if stats else None)
            if no_speech:
                latency["total_ms"] = _ms(t_total_start)
                logger.info("[VisualChat] No speech (%s) — transcript %r, voiced %.2fs — skipping turn.",
                            reason, text, stats.voiced_sec if stats else 0.0)
                if emotion_logger:
                    emotion_logger.log_skipped_turn(
                        session_id=session_id, transcript=text, reason=reason,
                        voiced_sec=stats.voiced_sec if stats else None, latency=latency,
                    )
                return VisualChatResponse(
                    transcript=text, no_speech=True, no_speech_reason=reason, latency=latency,
                )

        # ── Step 3: experts in parallel + fusion ─────────────────────
        analysis = await _analyse_turn(
            request, session_id, text,
            speech_waveform=speech_waveform, images=image_bytes,
        )
        latency.update(analysis["latency"])
        fused = analysis["fused"]

        # ── Step 4: LLM ───────────────────────────────────────────────
        async def generate(**kwargs: Any) -> Tuple[str, str]:
            return await llm_service.generate_visual_chat_response(
                provider=llm_provider, api_key=llm_api_key, model=llm_model, **kwargs,
            )

        t_llm = time.perf_counter()
        llm_emotion, llm_sentence = await _llm_turn(
            request, session_id, text, analysis, generate,
            lambda r: (r[1], r[0]),
        )
        latency["llm_ms"] = _ms(t_llm)
        logger.info("/chat/visual: LLM → emotion=%s | sentence=%r (llm_ms=%.0f)",
                    llm_emotion, llm_sentence[:60], latency["llm_ms"])

        dsmr_result = analysis["dsmr_result"]
        avatar_commands = {
            **(dsmr_result.get("avatar_commands") or {}),
            "expression": llm_emotion,
        }

        # ── Step 5: research logging ─────────────────────────────────
        turn_id: Optional[str] = None
        latency["total_ms"] = _ms(t_total_start)
        if emotion_logger:
            try:
                turn_id = emotion_logger.log_turn(
                    session_id=session_id,
                    transcript=text,
                    text_emotion=analysis["text"],
                    audio_emotion=analysis["audio"],
                    face_emotion=analysis["face"],
                    dsmr_emotion=analysis["dsmr"],
                    llm_response_emotion=llm_emotion,
                    llm_response_text=llm_sentence,
                    latency=latency,
                    fused_emotion=fused,
                )
            except Exception as e:
                logger.warning("/chat/visual: emotion_logger failed: %s", e)

        audio_success = True
        return VisualChatResponse(
            transcript=text,
            text_emotion=analysis["text"],
            audio_emotion=analysis["audio"],
            face_emotion=analysis["face"],
            dsmr_emotion=analysis["dsmr"],
            fused_emotion=fused,
            llm_emotion=llm_emotion,
            llm_sentence=llm_sentence,
            emotion_state=dsmr_result["emotion_state"],
            emotion_embedding=dsmr_result["emotion_embedding"],
            voice_params=response_voice_params(llm_emotion),
            avatar_commands=avatar_commands,
            audio_url=f"/uploads/audio/{audio_path.name}" if audio_path is not None else None,
            turn_id=turn_id,
            latency=latency,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("/chat/visual endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Visual chat pipeline error: {exc}")

    finally:
        if not audio_success:
            _safe_cleanup(audio_path)
