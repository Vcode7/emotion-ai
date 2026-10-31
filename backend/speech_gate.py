"""
speech_gate.py — Decide whether an utterance actually contains speech
=====================================================================

The browser VAD fires on clicks, coughs and the tail of the AI's own TTS,
and Whisper then hallucinates text for that near-silence ("Thank you.",
"you", "Bye."). In the research log ~45% of turns were such phantom turns,
and the AI replied to every one of them.

This module provides:
    • ``analyse_speech``  — voiced duration + speech bounds from frame RMS.
    • ``trim_silence``    — cut idle pre-roll / trailing silence before SER/DMESR.
    • ``is_no_speech``    — combine transcript, Whisper confidences and
                            acoustic evidence into a single gate decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

# ─────────────────────────────────────────────────────────────────────────────
#  Tunables
# ─────────────────────────────────────────────────────────────────────────────

FRAME_SEC = 0.025
"""Analysis frame length (25 ms, non-overlapping)."""

ABS_MIN_RMS = 0.01
"""Frames quieter than this are never counted as voiced."""

MIN_VOICED_SEC = 0.3
"""Utterances with less voiced audio than this are treated as no speech."""

HALLUCINATION_MAX_VOICED_SEC = 0.8
"""A hallucination-like transcript is only trusted with more voiced audio than this."""

NO_SPEECH_PROB = 0.6
AVG_LOGPROB = -1.0
"""Whisper's own silence rule: no_speech_prob > 0.6 and avg_logprob < -1.0."""

HALLUCINATION_MEAN_NO_SPEECH_PROB = 0.2
"""Mean segment no_speech_prob above which a hallucination-like transcript is dropped."""

# Whisper's well-known outputs on silence, noise and echo (normalised form).
_HALLUCINATIONS = frozenset({
    "thank you", "thank you very much", "thank you so much", "thanks",
    "thanks for watching", "thank you for watching", "thanks for listening",
    "thank you for listening", "please subscribe", "like and subscribe",
    "you", "bye", "bye bye", "goodbye", "oh", "so", "uh", "um", "hmm", "mm",
    "the end", "amen", "subtitles by the amaraorg community",
})

_EMPTY_MARKERS = frozenset({"", "inaudible", "transcription unavailable"})


@dataclass
class SpeechStats:
    voiced_sec: float
    total_sec: float
    start_sec: Optional[float]
    end_sec: Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
#  Acoustic analysis
# ─────────────────────────────────────────────────────────────────────────────

def _frame_rms(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    w = waveform.detach().flatten().float().cpu()
    frame = max(1, int(sr * FRAME_SEC))
    n = w.numel() // frame
    if n == 0:
        return torch.zeros(0)
    frames = w[: n * frame].view(n, frame)
    return frames.pow(2).mean(dim=1).sqrt()


def _voiced_mask(rms: torch.Tensor) -> torch.Tensor:
    """Adaptive threshold: well above the noise floor, but reachable by the loud frames."""
    if rms.numel() == 0:
        return torch.zeros(0, dtype=torch.bool)
    noise_floor = torch.quantile(rms, 0.2).item()
    peak = torch.quantile(rms, 0.95).item()
    threshold = max(ABS_MIN_RMS, min(3.0 * noise_floor, 0.5 * peak))
    return rms >= threshold


def analyse_speech(waveform: torch.Tensor, sr: int) -> SpeechStats:
    """Return voiced duration and the first/last voiced time of *waveform*."""
    rms = _frame_rms(waveform, sr)
    total_sec = waveform.numel() / float(sr) if sr else 0.0
    mask = _voiced_mask(rms)
    voiced_idx = torch.nonzero(mask).flatten()
    if voiced_idx.numel() == 0:
        return SpeechStats(voiced_sec=0.0, total_sec=total_sec, start_sec=None, end_sec=None)
    return SpeechStats(
        voiced_sec=round(voiced_idx.numel() * FRAME_SEC, 3),
        total_sec=round(total_sec, 3),
        start_sec=round(voiced_idx[0].item() * FRAME_SEC, 3),
        end_sec=round((voiced_idx[-1].item() + 1) * FRAME_SEC, 3),
    )


def trim_silence(waveform: torch.Tensor, sr: int, pad_sec: float = 0.2,
                 stats: Optional[SpeechStats] = None) -> torch.Tensor:
    """
    Cut leading / trailing silence (keeping *pad_sec* of context).

    The VAD recorder blob can contain up to 30 s of idle pre-roll, which
    dilutes speech-emotion models. Returns the input unchanged when no
    voiced region is found.
    """
    stats = stats or analyse_speech(waveform, sr)
    if stats.start_sec is None or stats.end_sec is None:
        return waveform
    flat = waveform.flatten()
    start = max(0, int((stats.start_sec - pad_sec) * sr))
    end = min(flat.numel(), int((stats.end_sec + pad_sec) * sr))
    if end - start < int(0.1 * sr):
        return waveform
    return flat[start:end]


# ─────────────────────────────────────────────────────────────────────────────
#  Gate decision
# ─────────────────────────────────────────────────────────────────────────────

def normalise_transcript(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    cleaned = re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower())
    cleaned = cleaned.replace("'", "")
    return re.sub(r"\s+", " ", cleaned).strip()


def is_hallucination_phrase(text: str) -> bool:
    return normalise_transcript(text) in _HALLUCINATIONS


def is_no_speech(
    text: str,
    segments: Optional[List[Dict[str, Any]]] = None,
    voiced_sec: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Decide whether a transcribed utterance should be ignored.

    Args:
        text:       Transcript from Whisper / Groq.
        segments:   Whisper segments; may carry ``no_speech_prob`` / ``avg_logprob``.
        voiced_sec: Voiced duration from :func:`analyse_speech` (``None`` = unknown).

    Returns:
        ``(no_speech, reason)`` — reason is empty when speech is accepted.
    """
    norm = normalise_transcript(text)
    if norm in _EMPTY_MARKERS:
        return True, "empty_transcript"

    if voiced_sec is not None and voiced_sec < MIN_VOICED_SEC:
        return True, f"too_little_voice ({voiced_sec:.2f}s)"

    scored = [
        s for s in (segments or [])
        if s.get("no_speech_prob") is not None
    ]
    if scored and all(
        float(s["no_speech_prob"]) > NO_SPEECH_PROB
        and float(s.get("avg_logprob", 0.0)) < AVG_LOGPROB
        for s in scored
    ):
        return True, "whisper_no_speech"

    if norm in _HALLUCINATIONS:
        mean_nsp = (
            sum(float(s["no_speech_prob"]) for s in scored) / len(scored)
            if scored else None
        )
        weak_voice = voiced_sec is not None and voiced_sec < HALLUCINATION_MAX_VOICED_SEC
        unsure = mean_nsp is not None and mean_nsp > HALLUCINATION_MEAN_NO_SPEECH_PROB
        if weak_voice or unsure:
            return True, "hallucination_phrase"

    return False, ""
