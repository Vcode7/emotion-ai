"""
emotion_fusion.py — Late fusion of text / voice / face emotion signals
======================================================================

Each expert model speaks its own vocabulary (GoEmotions 28 labels, SUPERB
4 labels, 8 facial-expression labels). This module projects them onto one
shared core label space, weights each modality by a base reliability and
its own confidence, and returns a single user-emotion read together with
per-modality evidence, an agreement score and a words-vs-nonverbal
conflict flag.

DMESR only predicts a sentiment scalar, so it is used as a gentle valence
prior rather than as a categorical vote.

    fused = fuse(text=text_result, audio=audio_result, face=face_result,
                 dsmr_valence=-0.24, user_text="I'm fine")
    fused["emotion"], fused["confidence"], fused["conflict"]
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

CORE_LABELS: Tuple[str, ...] = ("neutral", "happy", "sad", "angry", "fearful", "surprised")
NEGATIVE_LABELS = frozenset({"sad", "angry", "fearful"})

# Base reliability per modality (before scaling by each read's own confidence).
# Text is the most reliable on conversational speech; the face model is biased
# toward "sad" on resting faces and the voice model toward "angry".
BASE_WEIGHTS: Dict[str, float] = {"text": 0.45, "face": 0.30, "audio": 0.25}

DMESR_PRIOR_STRENGTH = 0.3
"""How strongly DMESR sentiment nudges the fused distribution."""

FACE_CONFLICT_MIN = 0.6
VOICE_CONFLICT_MIN = 0.85
"""Minimum non-verbal probability before a words-vs-nonverbal mismatch is flagged.
The voice model is only ~65% accurate (4 classes), so it needs to be very sure."""

POSITIVE_WORDS_MIN = 0.6
"""Words must read clearly positive (or be a minimising phrase) to contradict a negative face.
A neutral sentence ("Hello, how are you?") with a sad face is not a contradiction."""

FACE_BASELINE_MIN_OBS = 3
FACE_BASELINE_CORRECTION = 0.5

# Approximate valence / arousal of each core label, for a continuous summary.
_VA: Dict[str, Tuple[float, float]] = {
    "neutral":   (0.0, 0.0),
    "happy":     (0.8, 0.5),
    "sad":       (-0.7, -0.3),
    "angry":     (-0.6, 0.7),
    "fearful":   (-0.6, 0.6),
    "surprised": (0.2, 0.7),
}

_GOEMOTIONS_TO_CORE: Dict[str, str] = {
    "admiration": "happy", "amusement": "happy", "approval": "happy",
    "caring": "happy", "desire": "happy", "excitement": "happy",
    "gratitude": "happy", "joy": "happy", "love": "happy",
    "optimism": "happy", "pride": "happy", "relief": "happy",
    "anger": "angry", "annoyance": "angry", "disapproval": "angry", "disgust": "angry",
    "disappointment": "sad", "embarrassment": "sad", "grief": "sad",
    "remorse": "sad", "sadness": "sad",
    "fear": "fearful", "nervousness": "fearful",
    "surprise": "surprised", "realization": "surprised",
    "confusion": "neutral", "curiosity": "neutral", "neutral": "neutral",
}

_AUDIO_TO_CORE: Dict[str, str] = {
    "neu": "neutral", "neutral": "neutral",
    "hap": "happy", "happy": "happy", "excited": "happy",
    "ang": "angry", "angry": "angry", "frustrated": "angry",
    "sad": "sad",
}

_FACE_TO_CORE: Dict[str, str] = {
    "neutral": "neutral", "contempt": "neutral",
    "happy": "happy", "happiness": "happy",
    "sad": "sad", "sadness": "sad",
    "anger": "angry", "disgust": "angry",
    "fear": "fearful",
    "surprise": "surprised",
}

# Whole-phrase minimising statements ("I'm fine", "nothing's wrong").
# Word boundaries matter: plain substring matching turned "Good luck." into
# "You say you're good luck, but you look a little sad".
_MINIMISING_RE = re.compile(
    r"\b(?:i'?m|i am|im)\s+(?:fine|okay|ok|alright|all right|good|great|totally fine)\b"
    r"|\bnothing'?s?\s+(?:is\s+)?wrong\b"
    r"|\b(?:it'?s|its)\s+nothing\b"
    r"|\ball good\b|\bno problem\b|\bnot bad\b|\bjust tired\b|\bdon'?t worry\b"
    r"|^\s*(?:fine|okay|ok|alright|good)\s*[.!]?\s*$",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Projection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(dist: Dict[str, float]) -> Dict[str, float]:
    total = sum(dist.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in dist.items()}


def to_core(probabilities: Dict[str, float], mapping: Dict[str, str]) -> Dict[str, float]:
    """Sum raw-label probabilities into the core label space and normalise."""
    core = {label: 0.0 for label in CORE_LABELS}
    for raw, p in (probabilities or {}).items():
        target = mapping.get(str(raw).lower())
        if target:
            core[target] += max(0.0, float(p))
    return _normalise(core)


def _top(dist: Dict[str, float]) -> Tuple[str, float]:
    if not dist:
        return "neutral", 0.0
    label = max(dist, key=dist.get)
    return label, dist[label]


def text_nuance(probabilities: Dict[str, float], k: int = 3, min_p: float = 0.1) -> List[str]:
    """Top raw GoEmotions labels (excluding neutral) — e.g. ['gratitude', 'caring']."""
    ranked = sorted(
        ((lbl, p) for lbl, p in (probabilities or {}).items() if lbl != "neutral" and p >= min_p),
        key=lambda x: x[1], reverse=True,
    )
    return [lbl for lbl, _ in ranked[:k]]


def calibrate_face(dist: Dict[str, float], baseline: Optional[Dict[str, float]],
                   observations: int) -> Dict[str, float]:
    """
    Remove part of this user's habitual (resting-face) bias.

    Non-neutral labels that the face model reports more often than chance
    over the session are reduced by half of that excess, so a resting face
    that the model reads as "sad" does not make the user permanently sad.
    """
    if not dist or not baseline or observations < FACE_BASELINE_MIN_OBS:
        return dist
    chance = 1.0 / len(CORE_LABELS)
    adjusted = {}
    for label, p in dist.items():
        if label == "neutral":
            adjusted[label] = p
            continue
        excess = max(0.0, baseline.get(label, 0.0) - chance)
        adjusted[label] = max(0.0, p - FACE_BASELINE_CORRECTION * excess)
    return _normalise(adjusted) or dist


def sentiment_band(valence: Optional[float]) -> str:
    """Describe a DMESR valence in [-1, 1] as a sentiment band."""
    if valence is None:
        return "unknown"
    if valence <= -0.5:
        return "negative"
    if valence <= -0.15:
        return "slightly negative"
    if valence < 0.15:
        return "neutral"
    if valence < 0.5:
        return "slightly positive"
    return "positive"


def dsmr_sentiment(emotion_state: Dict[str, float]) -> Dict[str, Any]:
    """Present DMESR output honestly: it is a sentiment / valence estimate."""
    valence = float(emotion_state.get("valence", 0.0))
    return {
        "emotion": sentiment_band(valence),
        "valence": round(valence, 3),
        # DMESR has no class probabilities; expose sentiment intensity instead.
        "confidence": round(min(1.0, abs(valence)), 3),
        "kind": "sentiment",
        "emotion_state": emotion_state,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Fusion
# ─────────────────────────────────────────────────────────────────────────────

def _modality_read(name: str, result: Optional[Dict[str, Any]],
                   face_baseline: Optional[Dict[str, float]] = None,
                   face_observations: int = 0) -> Dict[str, Any]:
    """Project one expert result onto the core space and decide its fusion quality."""
    read: Dict[str, Any] = {"available": False, "used": False, "emotion": None,
                            "confidence": 0.0, "distribution": {}, "note": ""}
    if not result or result.get("available") is False:
        read["note"] = "unavailable"
        return read
    if name == "face" and result.get("face_detected") is False:
        read["note"] = "no face detected"
        return read

    mapping = {"text": _GOEMOTIONS_TO_CORE, "audio": _AUDIO_TO_CORE, "face": _FACE_TO_CORE}[name]
    dist = to_core(result.get("probabilities") or {}, mapping)
    if not dist:
        # No probabilities supplied — fall back to the service's own label.
        label = str(result.get("emotion") or "neutral").lower()
        label = label if label in CORE_LABELS else "neutral"
        conf = float(result.get("confidence") or 0.0)
        dist = {lbl: (conf if lbl == label else (1 - conf) / (len(CORE_LABELS) - 1)) for lbl in CORE_LABELS}

    read["available"] = True
    read["raw_distribution"] = dist
    if name == "face":
        calibrated = calibrate_face(dist, face_baseline, face_observations)
        if calibrated is not dist:
            read["note"] = "calibrated to this user's resting face"
        dist = calibrated

    label, conf = _top(dist)
    read.update({"emotion": label, "confidence": round(conf, 4),
                 "distribution": {k: round(v, 4) for k, v in dist.items()}})

    if result.get("gated"):
        read["note"] = (read["note"] + "; " if read["note"] else "") + "low confidence"
        return read  # too uncertain to vote
    read["used"] = True
    return read


def detect_conflict(user_text: str, text_read: Dict[str, Any], face_read: Dict[str, Any],
                    audio_read: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Flag a mismatch between what the user says and their non-verbal signals.

    Requires an explicit minimising phrase ("I'm fine") or clearly positive
    words, AND a confident negative face or voice read.
    """
    minimising = bool(_MINIMISING_RE.search(user_text or ""))
    words_positive = (
        text_read.get("used")
        and text_read.get("emotion") == "happy"
        and text_read.get("confidence", 0.0) >= POSITIVE_WORDS_MIN
    )
    if not (minimising or words_positive):
        return None

    candidates = []
    if face_read.get("used") and face_read.get("emotion") in NEGATIVE_LABELS \
            and face_read.get("confidence", 0.0) >= FACE_CONFLICT_MIN:
        candidates.append(("face", face_read))
    if audio_read.get("used") and audio_read.get("emotion") in NEGATIVE_LABELS \
            and audio_read.get("confidence", 0.0) >= VOICE_CONFLICT_MIN:
        candidates.append(("voice", audio_read))
    if not candidates:
        return None

    source, read = max(candidates, key=lambda c: c[1]["confidence"])
    return {
        "said": "minimising" if minimising else "positive",
        "observed": read["emotion"],
        "source": source,
        "confidence": read["confidence"],
        "minimising_phrase": minimising,
    }


def fuse(
    text: Optional[Dict[str, Any]] = None,
    audio: Optional[Dict[str, Any]] = None,
    face: Optional[Dict[str, Any]] = None,
    dsmr_valence: Optional[float] = None,
    user_text: str = "",
    face_baseline: Optional[Dict[str, float]] = None,
    face_observations: int = 0,
) -> Dict[str, Any]:
    """
    Fuse expert outputs into one user-emotion read.

    Returns a dict with: emotion, confidence, distribution, valence, arousal,
    agreement ("n/m"), agreement_ratio, modalities{text,audio,face}, nuance,
    conflict (dict or None), dsmr_valence.
    """
    reads = {
        "text": _modality_read("text", text),
        "audio": _modality_read("audio", audio),
        "face": _modality_read("face", face, face_baseline, face_observations),
    }

    fused = {label: 0.0 for label in CORE_LABELS}
    total_weight = 0.0
    for name, read in reads.items():
        if not read["used"]:
            continue
        weight = BASE_WEIGHTS[name] * read["confidence"]
        read["weight"] = round(weight, 4)
        total_weight += weight
        for label, p in read["distribution"].items():
            fused[label] += weight * p

    if total_weight <= 0:
        fused = {label: (1.0 if label == "neutral" else 0.0) for label in CORE_LABELS}
        confidence_scale = 0.0
    else:
        fused = _normalise(fused)
        confidence_scale = 1.0
        if dsmr_valence is not None:
            pos, neg = max(dsmr_valence, 0.0), max(-dsmr_valence, 0.0)
            fused["happy"] *= 1 + DMESR_PRIOR_STRENGTH * pos
            for label in NEGATIVE_LABELS:
                fused[label] *= 1 + DMESR_PRIOR_STRENGTH * neg
            fused = _normalise(fused)

    label, conf = _top(fused)
    used = [n for n, r in reads.items() if r["used"]]
    agreeing = [n for n in used if reads[n]["emotion"] == label]
    valence = sum(fused[l] * _VA[l][0] for l in CORE_LABELS)
    arousal = sum(fused[l] * _VA[l][1] for l in CORE_LABELS)

    return {
        "emotion": label,
        "confidence": round(conf * confidence_scale, 4),
        "distribution": {k: round(v, 4) for k, v in fused.items()},
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "agreement": f"{len(agreeing)}/{len(used)}",
        "agreement_ratio": round(len(agreeing) / len(used), 3) if used else 0.0,
        "modalities": {
            name: {k: v for k, v in read.items() if k != "raw_distribution"}
            for name, read in reads.items()
        },
        "face_raw_distribution": reads["face"].get("raw_distribution"),
        "nuance": text_nuance((text or {}).get("probabilities") or {}),
        "conflict": detect_conflict(user_text, reads["text"], reads["face"], reads["audio"]),
        "dsmr_valence": None if dsmr_valence is None else round(dsmr_valence, 3),
    }
