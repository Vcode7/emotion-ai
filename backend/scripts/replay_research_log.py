"""
replay_research_log.py — Re-run the new gating / conflict / parsing logic over
the research log and compare with what the old pipeline did.

    python scripts/replay_research_log.py [path/to/emotion_research_log.jsonl]

Only per-modality labels and confidences are logged (not full probabilities),
and no audio is stored, so this is an approximation:
  • phantom turns   — transcripts the speech gate treats as empty or as
                      Whisper hallucinations (dropped when voice evidence is weak),
  • conflicts       — old substring heuristic vs new fusion-based detection,
  • leaked replies  — stored AI replies that still contain "emotion(" syntax.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from emotion_fusion import CORE_LABELS, fuse  # noqa: E402
from llm_service import _parse_emotion_sentence  # noqa: E402
from speech_gate import is_hallucination_phrase, normalise_transcript  # noqa: E402

_OLD_MINIMISING = [
    "i'm fine", "im fine", "i am fine", "fine",
    "i'm okay", "im okay", "i am okay", "okay", "ok", "i'm ok", "im ok",
    "nothing", "nothing is wrong", "nothing's wrong",
    "all good", "i'm good", "im good", "good",
    "i'm alright", "im alright", "alright", "all right",
    "doing well", "no problem", "not bad", "just tired",
]
_OLD_DISTRESS = {"sad", "sadness", "angry", "anger", "fear", "fearful", "disgust", "concerned"}
_TO_CORE = {"empathetic": "happy", "thinking": "neutral", "concerned": "fearful"}


def old_conflict(rec: dict) -> bool:
    face = rec["face_emotion"]
    if not (face.get("available") and face.get("face_detected")):
        return False
    if face.get("emotion") not in _OLD_DISTRESS:
        return False
    text = rec["transcript"].lower()
    return any(p in text for p in _OLD_MINIMISING)


def as_result(d: dict, face: bool = False) -> dict:
    label = _TO_CORE.get(d.get("emotion"), d.get("emotion"))
    label = label if label in CORE_LABELS else "neutral"
    out = {"available": d.get("available", False), "emotion": label, "confidence": d.get("confidence", 0.0)}
    if face:
        out["face_detected"] = d.get("face_detected", False)
    return out


def main(path: Path) -> None:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    turns = [r for r in rows if not r.get("no_speech") and "text_emotion" in r]
    print(f"Log: {path}  ({len(turns)} answered turns, {len(rows) - len(turns)} already-skipped)")

    phantom = [r for r in turns
               if normalise_transcript(r["transcript"]) in ("", "inaudible")
               or is_hallucination_phrase(r["transcript"])]
    print(f"\nPhantom-turn candidates: {len(phantom)} / {len(turns)} "
          f"({100 * len(phantom) / max(1, len(turns)):.0f}%) — gated when voice evidence is weak")
    print("  most common:", Counter(r["transcript"] for r in phantom).most_common(6))

    real = [r for r in turns if r not in phantom]
    old_hits, new_hits = [], {}
    for r in real:
        fused = fuse(text=as_result(r["text_emotion"]), audio=as_result(r["audio_emotion"]),
                     face=as_result(r["face_emotion"], face=True), user_text=r["transcript"])
        if old_conflict(r):
            old_hits.append(r)
        if fused["conflict"]:
            new_hits[id(r)] = fused["conflict"]
    print(f"\nWords-vs-nonverbal conflicts on real turns: old heuristic {len(old_hits)}, new fusion {len(new_hits)}")
    for r in old_hits:
        if id(r) not in new_hits:
            f = r["face_emotion"]
            print(f"  no longer flagged: {r['transcript']!r} (face {f['emotion']} {f['confidence']:.2f})")
    for r in real:
        c = new_hits.get(id(r))
        if c:
            print(f"  flagged:           {r['transcript']!r} ({c['source']} {c['observed']} {c['confidence']:.2f})")

    leaked = [r for r in turns if re.match(r"^\s*[A-Za-z_]+\s*\(", r.get("llm_response_text", ""))]
    print(f"\nStored replies with leaked emotion(...) syntax: {len(leaked)}")
    for r in leaked:
        print(f"  {r['llm_response_text'][:60]!r} → now parsed as {_parse_emotion_sentence(r['llm_response_text'])}")


if __name__ == "__main__":
    default = BACKEND_DIR / "logs" / "emotion_research_log.jsonl"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
