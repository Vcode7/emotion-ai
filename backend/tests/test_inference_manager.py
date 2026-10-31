"""
InferenceManager: shared BERT across sessions, LRU eviction, timestamps.

Uses the real DMESR model + checkpoint; only the frozen BERT is faked so the
test does not download bert-base-uncased.
"""

from pathlib import Path

import pytest
import torch

import inference

CKPT = Path(inference._PROJECT_ROOT) / "model" / "checkpoint" / "dmesr_mosei_best.pt"


class FakeTokenizer:
    def __call__(self, text, padding, truncation, max_length, return_tensors):
        return {"input_ids": torch.zeros(1, max_length, dtype=torch.long),
                "attention_mask": torch.ones(1, max_length, dtype=torch.long)}


class FakeBert:
    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, attention_mask=None):
        self.calls += 1
        torch.manual_seed(0)
        return type("Out", (), {"last_hidden_state": torch.randn(1, input_ids.shape[1], 768)})()


@pytest.mark.skipif(not CKPT.exists(), reason="DMESR checkpoint not present")
def test_sessions_share_bert_and_are_lru_capped(monkeypatch):
    loads = []

    def fake_loader(device="cpu"):
        loads.append(device)
        return FakeTokenizer(), FakeBert()

    monkeypatch.setattr(inference, "load_frozen_bert", fake_loader)
    manager = inference.InferenceManager(device="cpu", max_sessions=2)

    out = manager.process_text_only("a", "I am happy")
    manager.process_text_only("b", "hello")
    manager.process_text_only("a", "again")        # touch "a" → "b" is now LRU
    manager.process_text_only("c", "new session")  # evicts "b"

    assert len(loads) == 1                          # BERT loaded once, not per session
    assert manager.session_count == 2
    assert set(manager._sessions) == {"a", "c"}
    assert manager.get_session("a").bert is manager.get_session("c").bert
    assert out["timestamp"]                          # timeline timestamps are filled
    assert set(out["emotion_state"]) >= {"valence", "arousal"}
