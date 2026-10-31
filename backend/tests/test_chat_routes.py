"""
Route-level tests for routes/chat.py with fake model services.

The real endpoint code, speech gate, fusion, prompts, conversation memory
and parsers run; only the heavy models and the LLM network call are faked.
"""

import io
import math
import wave

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

import llm_service
import routes.chat as chat
from conversation_store import ConversationStore

SR = 16_000


# ── Fakes ────────────────────────────────────────────────────────────────────

class FakeText:
    available = True

    def analyse(self, text):
        t = text.lower()
        if "lost" in t or "sad" in t:
            probs = {"sadness": 0.8, "neutral": 0.2}
        elif "fine" in t:
            probs = {"neutral": 0.85, "approval": 0.15}
        else:
            probs = {"neutral": 0.9, "joy": 0.1}
        top = max(probs, key=probs.get)
        return {"emotion": top, "raw_emotion": top, "confidence": probs[top],
                "probabilities": probs, "available": True}


class FakeAudio:
    available = True

    def __init__(self):
        self.calls = 0

    def analyse(self, waveform):
        self.calls += 1
        return {"emotion": "neutral", "confidence": 0.8, "probabilities": {"neu": 0.8, "sad": 0.2},
                "gated": False, "available": True}


class FakeFace:
    available = True

    def __init__(self, probs=None, explode=False):
        self.probs = probs or {"neutral": 0.9, "sad": 0.1}
        self.explode = explode
        self.frames_seen = []

    def analyse_many(self, images):
        if self.explode:
            raise RuntimeError("face model crashed")
        self.frames_seen.append(len(images))
        top = max(self.probs, key=self.probs.get)
        return {"emotion": top, "confidence": self.probs[top], "probabilities": self.probs,
                "face_detected": True, "gated": False, "available": True}


class FakeWhisper:
    available = True

    def __init__(self, text, segments=()):
        self.text = text
        self.segments = list(segments)

    def transcribe(self, path):
        return {"text": self.text, "segments": list(self.segments)}


class FakeInference:
    def __init__(self):
        self.calls = []

    def _result(self, text):
        state = {"valence": -0.2, "arousal": 0.2}
        return {"emotion_state": state, "emotion_embedding": [0.0] * 4,
                "llm_context": {"recent_history": [{"sentence": text, "emotion_state": state, "timestamp": "t"}]},
                "avatar_commands": {"expression": "joyful", "smile": 0.3}, "voice_params": {}}

    def process_text_only(self, sid, text):
        self.calls.append(("text", sid))
        return self._result(text)

    def process_with_audio(self, sid, text, wave_):
        self.calls.append(("audio", sid))
        return self._result(text)

    def process_with_video(self, sid, text, video):
        return self._result(text)

    def process_full(self, sid, text, wave_, video):
        return self._result(text)


class FakeLogger:
    def __init__(self):
        self.turns, self.skipped, self.simple = [], [], []

    def log_turn(self, **kw):
        self.turns.append(kw)
        return "turn-1"

    def log_skipped_turn(self, **kw):
        self.skipped.append(kw)

    def log_simple_turn(self, **kw):
        self.simple.append(kw)


def wav_bytes(samples):
    pcm = (samples.clamp(-1, 1) * 32767).to(torch.int16).numpy().tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    return buf.getvalue()


def speech_like(seconds=2.0):
    t = torch.arange(int(seconds * SR)) / SR
    voiced = 0.3 * torch.sin(2 * math.pi * 220 * t)
    pad = 0.0005 * torch.randn(int(0.8 * SR))
    return torch.cat([pad, voiced, pad])


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def llm_calls(monkeypatch):
    calls = []

    async def fake_groq(messages, api_key, model, max_tokens=250):
        calls.append(messages)
        if "emotion(Your response sentence here)" in messages[0]["content"]:
            return "<think>keep it short</think>\nempathetic(I'm right here with you.)"
        return 'Sure:\n[{"text": "I hear you.", "emotion": "compassion"}]'

    monkeypatch.setitem(llm_service._CALLERS, "groq", fake_groq)
    return calls


@pytest.fixture
def make_client(monkeypatch, tmp_path):
    monkeypatch.setattr(chat, "UPLOAD_AUDIO_DIR", tmp_path)

    def _make(whisper_text="", segments=(), face=None):
        app = FastAPI()
        app.include_router(chat.router)
        s = app.state
        s.inference_manager = FakeInference()
        s.llm_service = llm_service.LLMService()
        s.whisper_service = FakeWhisper(whisper_text, segments)
        s.groq_whisper_service = None
        s.text_emotion_service = FakeText()
        s.audio_emotion_service = FakeAudio()
        s.face_emotion_service = face or FakeFace()
        s.emotion_logger = FakeLogger()
        s.conversation_store = ConversationStore()
        return TestClient(app), s

    return _make


VISUAL_FORM = {"session_id": "s1", "llm_provider": "groq", "llm_api_key": "test-key", "stt_provider": "whisper"}


# ── /chat/visual ─────────────────────────────────────────────────────────────

def test_visual_text_only_no_longer_500s(make_client, llm_calls):
    client, _ = make_client()
    r = client.post("/chat/visual", data={**VISUAL_FORM, "user_text": "I lost my job today"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["audio_url"] is None
    assert body["llm_emotion"] == "empathetic"
    assert body["llm_sentence"] == "I'm right here with you."
    assert body["fused_emotion"]["emotion"] == "sad"
    assert body["dsmr_emotion"]["emotion"] == "slightly negative"
    assert body["voice_params"]["emotion"] == "empathetic"


def test_visual_silence_with_thank_you_hallucination_is_skipped(make_client, llm_calls, tmp_path):
    client, state = make_client(whisper_text="Thank you.")
    silent = wav_bytes(0.0005 * torch.randn(3 * SR))
    r = client.post("/chat/visual", data=VISUAL_FORM,
                    files={"audio": ("recording.wav", silent, "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["no_speech"] is True
    assert body["llm_sentence"] == ""
    assert llm_calls == []                               # no LLM call
    assert state.inference_manager.calls == []           # DMESR memory untouched
    assert len(state.emotion_logger.skipped) == 1
    assert list(tmp_path.iterdir()) == []                # upload cleaned up


def test_visual_real_speech_with_frames(make_client, llm_calls):
    face = FakeFace({"neutral": 0.7, "sad": 0.3})
    client, state = make_client(whisper_text="I had a long day", face=face)
    files = [
        ("audio", ("recording.wav", wav_bytes(speech_like()), "audio/wav")),
        ("images", ("f0.jpg", b"jpeg-bytes-0", "image/jpeg")),
        ("images", ("f1.jpg", b"jpeg-bytes-1", "image/jpeg")),
    ]
    r = client.post("/chat/visual", data=VISUAL_FORM, files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["no_speech"] is False
    assert body["audio_url"].startswith("/uploads/audio/")
    assert face.frames_seen == [2]
    assert state.audio_emotion_service.calls == 1
    assert state.inference_manager.calls == [("audio", "s1")]
    assert state.emotion_logger.turns[0]["fused_emotion"]["agreement"]


def test_visual_pipeline_error_is_reported_not_masked(make_client, llm_calls):
    client, _ = make_client(face=FakeFace(explode=True))
    r = client.post("/chat/visual", data={**VISUAL_FORM, "user_text": "hello"},
                    files={"images": ("f.jpg", b"x", "image/jpeg")})
    assert r.status_code == 500
    assert "face model crashed" in r.json()["detail"]


def test_visual_conflict_is_raised_once_then_cooled_down(make_client, llm_calls):
    client, _ = make_client(face=FakeFace({"sad": 0.85, "neutral": 0.15}))
    form = {**VISUAL_FORM, "user_text": "I'm fine"}
    frame = {"images": ("f.jpg", b"x", "image/jpeg")}
    client.post("/chat/visual", data=form, files=frame)
    client.post("/chat/visual", data=form, files=frame)
    first_sys, second_sys = llm_calls[0][0]["content"], llm_calls[1][0]["content"]
    assert "MISMATCH NOTICED" in first_sys
    assert "MISMATCH NOTICED" not in second_sys
    assert "Do not bring it up again" in second_sys


# ── /chat/text ───────────────────────────────────────────────────────────────

def test_text_chat_has_memory_and_fused_user_emotion(make_client, llm_calls):
    client, _ = make_client()
    body = {"session_id": "s2", "llm_provider": "groq", "llm_api_key": "test-key"}
    for msg in ("I lost my job today", "My manager was rude", "because of you."):
        r = client.post("/chat/text", json={**body, "text": msg})
        assert r.status_code == 200, r.text

    last = r.json()
    assert last["response"] == [{"text": "I hear you.", "emotion": "compassion"}]
    # Avatar/voice follow the AI reply, not DMESR's "joyful".
    assert last["avatar_commands"]["expression"] == "compassion"
    assert last["voice_params"]["emotion"] == "compassion"

    third_call = llm_calls[2]
    roles = [m["role"] for m in third_call]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert third_call[1]["content"] == "I lost my job today"
    assert third_call[-1]["content"] == "because of you."
    assert "Trajectory:" in third_call[0]["content"]

    first = client.post("/chat/text", json={**body, "session_id": "s3", "text": "I lost my job"}).json()
    assert first["user_emotion"] == "Sad"
    assert 0 < first["user_emotion_score"] <= 1


def test_audio_chat_no_speech_gets_canned_reply(make_client, llm_calls):
    client, _ = make_client(whisper_text="Thank you.")
    r = client.post(
        "/chat/audio",
        data={"session_id": "s4", "llm_provider": "groq", "llm_api_key": "k", "stt_provider": "whisper"},
        files={"audio": ("recording.wav", wav_bytes(0.0005 * torch.randn(2 * SR)), "audio/wav")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["no_speech"] is True
    assert body["response"][0]["text"] == chat.NO_SPEECH_REPLY
    assert llm_calls == []
