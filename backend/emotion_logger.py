"""
emotion_logger.py — Research-quality JSONL logging for multimodal emotion analysis
==================================================================================

Appends one JSON record per conversation turn to:
    backend/logs/emotion_research_log.jsonl

Each record captures the four emotion expert outputs, the LLM response,
and per-stage latency, enabling offline analysis of modality agreement,
disagreement, and LLM reasoning quality.

Log schema:
    {
        "timestamp":             ISO-8601 string,
        "session_id":            str,
        "turn_id":               str (UUID),
        "transcript":            str,

        "text_emotion":  {"emotion": str, "confidence": float, "available": bool},
        "audio_emotion": {"emotion": str, "confidence": float, "available": bool},
        "face_emotion":  {"emotion": str, "confidence": float, "available": bool, "face_detected": bool},
        "dsmr_emotion":  {"emotion": str (sentiment band), "confidence": float, "valence": float},
        "fused_emotion": {"emotion": str, "confidence": float, "agreement": str,
                          "conflict": {...} | null},

        "llm_response_emotion":  str,
        "llm_response_text":     str,

        "latency": {
            "stt_ms":          float | null,
            "text_emotion_ms": float | null,
            "audio_emotion_ms":float | null,
            "face_emotion_ms": float | null,
            "dsmr_ms":         float | null,
            "llm_ms":          float | null,
            "tts_ms":          float | null,
            "total_ms":        float | null,
        }
    }

Turns dropped by the speech gate are logged as
    {"timestamp", "session_id", "transcript", "no_speech": true,
     "skip_reason": str, "voiced_sec": float | null, "latency": {...}}
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOG_DIR       = Path(__file__).resolve().parent / "logs"
_LOG_FILE      = _LOG_DIR / "emotion_research_log.jsonl"
_CALL_LOG_FILE = _LOG_DIR / "call_summary.log"


class EmotionLogger:
    """
    Thread-safe append-only JSONL logger for multimodal emotion research
    and clean human-readable call summaries.

    Usage::

        emotion_logger = EmotionLogger()
        emotion_logger.log_turn(...)
        emotion_logger.log_call_session(...)
    """

    def __init__(self) -> None:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("EmotionLogger ready. Research: %s | Call Summary: %s", _LOG_FILE, _CALL_LOG_FILE)

    def log_turn(
        self,
        session_id: str,
        transcript: str,
        text_emotion: Dict[str, Any],
        audio_emotion: Dict[str, Any],
        face_emotion: Dict[str, Any],
        dsmr_emotion: Dict[str, Any],
        llm_response_emotion: str,
        llm_response_text: str,
        latency: Optional[Dict[str, Optional[float]]] = None,
        fused_emotion: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Append one turn record to the JSONL log file AND format a clear,
        human-readable call complete log to the console and call_summary.log.

        Returns the generated ``turn_id`` (UUID4 hex).
        """
        turn_id = uuid.uuid4().hex
        ts_now = datetime.now(timezone.utc)
        ts_str = ts_now.strftime("%Y-%m-%d %H:%M:%S UTC")

        # ── 1. Append to Research JSONL ──────────────────────────────
        record: Dict[str, Any] = {
            "timestamp":            ts_now.isoformat(),
            "session_id":           session_id,
            "turn_id":              turn_id,
            "transcript":           transcript,

            "text_emotion": _slim_emotion(text_emotion),
            "audio_emotion": _slim_emotion(audio_emotion),
            "face_emotion": _slim_face_emotion(face_emotion),
            "dsmr_emotion": _slim_dsmr(dsmr_emotion),
            "fused_emotion": _slim_fused(fused_emotion or {}),

            "llm_response_emotion": llm_response_emotion,
            "llm_response_text":    llm_response_text,

            "latency": _slim_latency(latency or {}),
        }

        try:
            with open(_LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("EmotionLogger: failed to write research record: %s", exc)

        # ── 2. Format Human-Readable Call Turn Log ────────────────────
        u_text_emo = text_emotion.get("emotion", "neutral")
        u_text_conf = round(float(text_emotion.get("confidence", 0.0)) * 100)

        u_audio_emo = audio_emotion.get("emotion", "neutral")
        u_audio_conf = round(float(audio_emotion.get("confidence", 0.0)) * 100)

        u_face_detected = face_emotion.get("face_detected", True)
        u_face_emo = face_emotion.get("emotion", "neutral") if u_face_detected else "No face detected"
        u_face_conf = round(float(face_emotion.get("confidence", 0.0)) * 100)

        u_dsmr_emo = dsmr_emotion.get("emotion", "neutral")
        u_dsmr_val = float(dsmr_emotion.get("valence", 0.0))

        fused = fused_emotion or {}
        u_fused_emo = fused.get("emotion", "neutral")
        u_fused_conf = round(float(fused.get("confidence", 0.0)) * 100)
        conflict = fused.get("conflict")
        conflict_str = (
            f"  • Conflict      : words vs {conflict.get('source')} ({conflict.get('observed')})\n"
            if conflict else ""
        )

        turn_banner = (
            "\n"
            "================================================================================\n"
            "                         📞 CALL COMPLETE LOG\n"
            "================================================================================\n"
            f"Timestamp        : {ts_str}\n"
            f"Session ID       : {session_id}\n"
            f"Turn ID          : {turn_id}\n"
            "--------------------------------------------------------------------------------\n"
            f"[USER SAID]      : \"{transcript}\"\n"
            "[USER EMOTIONS]  :\n"
            f"  • Text Emotion : {u_text_emo} ({u_text_conf}%)\n"
            f"  • Voice Emotion: {u_audio_emo} ({u_audio_conf}%)\n"
            f"  • Face Emotion : {u_face_emo} ({u_face_conf}%)\n"
            f"  • DSMR Sentiment: {u_dsmr_emo} (valence {u_dsmr_val:+.2f})\n"
            f"  • Overall (fused): {u_fused_emo} ({u_fused_conf}%, agree {fused.get('agreement', '-')})\n"
            f"{conflict_str}"
            "--------------------------------------------------------------------------------\n"
            f"[AI RESPONDED]   : \"{llm_response_text}\"\n"
            f"[AI EMOTION]     : {llm_response_emotion.upper()}\n"
            "================================================================================"
        )

        logger.info(turn_banner)

        try:
            with open(_CALL_LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(turn_banner + "\n\n")
        except OSError as exc:
            logger.error("EmotionLogger: failed to write call summary log: %s", exc)

        return turn_id

    def log_call_session(
        self,
        session_id: str,
        turns: List[Dict[str, Any]],
        duration_seconds: Optional[float] = None,
    ) -> None:
        """
        Log complete call session summary when user ends a call.
        """
        ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        dur_str = f"{duration_seconds:.1f}s" if duration_seconds is not None else "N/A"

        lines = [
            "\n"
            "================================================================================",
            "                     📞 CALL ENDED - FULL SESSION SUMMARY",
            "================================================================================",
            f"Timestamp        : {ts_str}",
            f"Session ID       : {session_id}",
            f"Call Duration    : {dur_str}",
            f"Total Turns      : {len(turns)}",
            "--------------------------------------------------------------------------------",
        ]

        if not turns:
            lines.append("  (No turns recorded during this call session)")
        else:
            for i, turn in enumerate(turns, start=1):
                user_said = turn.get("userSaid", turn.get("transcript", ""))
                user_emo = turn.get("userEmotion", {})
                if isinstance(user_emo, dict):
                    text_emo = f"{user_emo.get('text', 'neutral')} ({round(float(user_emo.get('textConfidence', 0)) * 100)}%)"
                    audio_emo = f"{user_emo.get('audio', 'neutral')} ({round(float(user_emo.get('audioConfidence', 0)) * 100)}%)"
                    face_emo = str(user_emo.get('face', 'neutral'))
                    dsmr_emo = str(user_emo.get('dsmr', 'neutral'))
                    overall = f"{user_emo.get('overall', 'neutral')} ({round(float(user_emo.get('overallConfidence', 0)) * 100)}%)"
                    emo_str = f"Text: {text_emo} | Voice: {audio_emo} | Face: {face_emo} | DSMR: {dsmr_emo} | Overall: {overall}"
                else:
                    emo_str = str(user_emo)

                ai_resp = turn.get("aiResponded", turn.get("llm_response_text", ""))
                ai_emo = str(turn.get("aiEmotion", turn.get("llm_response_emotion", "neutral"))).upper()

                lines.append(f"Turn #{i}:")
                lines.append(f"  [USER SAID]    : \"{user_said}\"")
                lines.append(f"  [USER EMOTION] : {emo_str}")
                lines.append(f"  [AI RESPONDED] : \"{ai_resp}\"")
                lines.append(f"  [AI EMOTION]   : {ai_emo}")
                if i < len(turns):
                    lines.append("  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

        lines.append("================================================================================")
        summary_text = "\n".join(lines)

        logger.info(summary_text)

        try:
            with open(_CALL_LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(summary_text + "\n\n")
        except OSError as exc:
            logger.error("EmotionLogger: failed to write session summary: %s", exc)

    def log_skipped_turn(
        self,
        session_id: str,
        transcript: str,
        reason: str,
        voiced_sec: Optional[float] = None,
        latency: Optional[Dict[str, Optional[float]]] = None,
    ) -> None:
        """Record a turn dropped by the speech gate (no LLM call was made)."""
        record = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "session_id":  session_id,
            "transcript":  transcript,
            "no_speech":   True,
            "skip_reason": reason,
            "voiced_sec":  voiced_sec,
            "latency":     _slim_latency(latency or {}),
        }
        try:
            with open(_LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("EmotionLogger: failed to write skipped-turn record: %s", exc)

    def log_simple_turn(
        self,
        session_id: str,
        user_text: str,
        user_emotion: str,
        user_score: float,
        ai_response_text: str,
        ai_response_emotion: str,
        channel: str = "chat",
    ) -> None:
        """
        Log a simple (text/audio) chat turn in proper call format.
        """
        ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        conf_str = f" ({round(user_score * 100)}%)" if user_score else ""
        turn_banner = (
            "\n"
            "================================================================================\n"
            f"                         💬 CHAT CALL COMPLETE LOG ({channel.upper()})\n"
            "================================================================================\n"
            f"Timestamp        : {ts_str}\n"
            f"Session ID       : {session_id}\n"
            "--------------------------------------------------------------------------------\n"
            f"[USER SAID]      : \"{user_text}\"\n"
            f"[USER EMOTION]   : {user_emotion}{conf_str}\n"
            "--------------------------------------------------------------------------------\n"
            f"[AI RESPONDED]   : \"{ai_response_text}\"\n"
            f"[AI EMOTION]     : {ai_response_emotion.upper()}\n"
            "================================================================================"
        )
        logger.info(turn_banner)
        try:
            with open(_CALL_LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(turn_banner + "\n\n")
        except OSError as exc:
            logger.error("EmotionLogger: failed to write simple turn log: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slim_emotion(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "emotion":    d.get("emotion", "neutral"),
        "confidence": round(float(d.get("confidence", 0.0)), 4),
        "available":  bool(d.get("available", False)),
    }


def _slim_face_emotion(d: Dict[str, Any]) -> Dict[str, Any]:
    base = _slim_emotion(d)
    base["face_detected"] = bool(d.get("face_detected", False))
    return base


def _slim_dsmr(d: Dict[str, Any]) -> Dict[str, Any]:
    slim = {
        "emotion":    d.get("emotion", "neutral"),
        "confidence": round(float(d.get("confidence", 0.0)), 4),
    }
    if d.get("valence") is not None:
        slim["valence"] = round(float(d["valence"]), 4)
    return slim


def _slim_fused(d: Dict[str, Any]) -> Dict[str, Any]:
    if not d:
        return {}
    return {
        "emotion":      d.get("emotion", "neutral"),
        "confidence":   round(float(d.get("confidence", 0.0)), 4),
        "agreement":    d.get("agreement"),
        "conflict":     d.get("conflict"),
        "distribution": d.get("distribution"),
    }


def _slim_latency(d: Dict[str, Any]) -> Dict[str, Optional[float]]:
    keys = [
        "stt_ms", "text_emotion_ms", "audio_emotion_ms",
        "face_emotion_ms", "dsmr_ms", "llm_ms", "tts_ms", "total_ms",
    ]
    return {k: (round(float(d[k]), 2) if d.get(k) is not None else None) for k in keys}
