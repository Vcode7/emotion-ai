"""
conversation_store.py — Per-session conversation memory and emotion state
=========================================================================

Keeps, per ``session_id``:
    • the last N conversation turns (user + assistant) for LLM context,
    • an EMA baseline of the user's facial-expression distribution
      (used to calibrate away a "sad-looking" resting face),
    • an EMA of the fused user emotion and the recent label trajectory,
    • when a words-vs-face mismatch was last raised, so the AI does not
      repeat "you say you're fine, but…" every turn.

Sessions are evicted least-recently-used beyond ``max_sessions`` and after
``ttl_seconds`` of inactivity.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from emotion_fusion import CORE_LABELS

FACE_BASELINE_ALPHA = 0.2
FUSED_EMA_ALPHA = 0.5
CONFLICT_COOLDOWN_TURNS = 3
LOW_CONFIDENCE = 0.4


@dataclass
class Turn:
    role: str                 # "user" | "assistant"
    text: str
    emotion: str = "neutral"
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionState:
    max_turns: int = 12
    turns: Deque[Turn] = field(default_factory=deque)
    face_baseline: Optional[Dict[str, float]] = None
    face_observations: int = 0
    fused_ema: Optional[Dict[str, float]] = None
    user_emotions: Deque[Tuple[str, float]] = field(default_factory=lambda: deque(maxlen=12))
    user_turn_count: int = 0
    last_conflict_turn: Optional[int] = None
    last_access: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.turns = deque(self.turns, maxlen=self.max_turns)

    # ── Conversation history ─────────────────────────────────────────────

    def add_user_turn(self, text: str, emotion: str, confidence: float) -> None:
        self.turns.append(Turn("user", text, emotion, confidence))
        self.user_emotions.append((emotion, confidence))
        self.user_turn_count += 1

    def add_assistant_turn(self, text: str, emotion: str) -> None:
        self.turns.append(Turn("assistant", text, emotion))

    def history_messages(self, last_n: int = 8) -> List[Dict[str, str]]:
        """Recent turns as chat messages (oldest first), excluding the current user turn."""
        recent = list(self.turns)[-last_n:]
        return [
            {"role": t.role, "content": t.text}
            for t in recent
            if t.text
        ]

    # ── Emotion state ────────────────────────────────────────────────────

    def update_face_baseline(self, distribution: Optional[Dict[str, float]]) -> None:
        """EMA of the raw (uncalibrated) face distribution over the session."""
        if not distribution:
            return
        if self.face_baseline is None:
            self.face_baseline = dict(distribution)
        else:
            self.face_baseline = {
                label: (1 - FACE_BASELINE_ALPHA) * self.face_baseline.get(label, 0.0)
                + FACE_BASELINE_ALPHA * distribution.get(label, 0.0)
                for label in CORE_LABELS
            }
        self.face_observations += 1

    def update_fused(self, distribution: Dict[str, float]) -> Dict[str, float]:
        """Update and return the EMA-smoothed fused distribution."""
        if self.fused_ema is None:
            self.fused_ema = dict(distribution)
        else:
            self.fused_ema = {
                label: (1 - FUSED_EMA_ALPHA) * self.fused_ema.get(label, 0.0)
                + FUSED_EMA_ALPHA * distribution.get(label, 0.0)
                for label in CORE_LABELS
            }
        return self.fused_ema

    def smoothed_emotion(self) -> Optional[Tuple[str, float]]:
        if not self.fused_ema:
            return None
        label = max(self.fused_ema, key=self.fused_ema.get)
        return label, round(self.fused_ema[label], 3)

    def trajectory(self) -> str:
        """Short description of how the user's emotion has moved over recent turns."""
        labels = [
            label if conf >= LOW_CONFIDENCE else "neutral"
            for label, conf in self.user_emotions
        ]
        if len(labels) < 2:
            return "first turn of the conversation" if labels else ""
        current = labels[-1]
        run = 1
        for label in reversed(labels[:-1]):
            if label != current:
                break
            run += 1
        if run >= 2:
            return f"{current} for the last {run} turns"
        return f"shifted from {labels[-2]} to {current}"

    # ── Conflict cooldown ────────────────────────────────────────────────

    def may_raise_conflict(self) -> bool:
        if self.last_conflict_turn is None:
            return True
        return self.user_turn_count - self.last_conflict_turn > CONFLICT_COOLDOWN_TURNS

    def mark_conflict_raised(self) -> None:
        self.last_conflict_turn = self.user_turn_count


class ConversationStore:
    """Thread-safe LRU + TTL store of :class:`SessionState` keyed by session id."""

    def __init__(self, max_sessions: int = 200, ttl_seconds: int = 3600, max_turns: int = 12) -> None:
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self._sessions: "OrderedDict[str, SessionState]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str) -> SessionState:
        now = time.time()
        with self._lock:
            self._evict_expired(now)
            state = self._sessions.get(session_id)
            if state is None:
                state = SessionState(max_turns=self.max_turns)
                self._sessions[session_id] = state
                while len(self._sessions) > self.max_sessions:
                    self._sessions.popitem(last=False)
            else:
                self._sessions.move_to_end(session_id)
            state.last_access = now
            return state

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._sessions)

    def _evict_expired(self, now: float) -> None:
        expired = [sid for sid, s in self._sessions.items() if now - s.last_access > self.ttl_seconds]
        for sid in expired:
            del self._sessions[sid]
