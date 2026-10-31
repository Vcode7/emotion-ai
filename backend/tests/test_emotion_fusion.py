import pytest

from conversation_store import HISTORY_MESSAGE_MAX_CHARS, HISTORY_MESSAGES, ConversationStore
from emotion_fusion import (
    calibrate_face, dsmr_sentiment, fuse, nonverbal_reads, sentiment_band, to_core, _GOEMOTIONS_TO_CORE,
)


def face(probs, detected=True, gated=False):
    return {"available": True, "face_detected": detected, "gated": gated, "probabilities": probs}


def text(probs):
    return {"available": True, "probabilities": probs}


def audio(probs, gated=False):
    return {"available": True, "gated": gated, "probabilities": probs}


# ── Conflict detection (regressions from the research log) ──────────────────

def test_good_luck_with_weak_sad_face_is_not_a_conflict():
    # Log: "Good luck." + face sad 0.50 → "You say you're good luck, but you look a little sad".
    fused = fuse(
        text=text({"neutral": 0.4, "caring": 0.5}),
        face=face({"sad": 0.50, "neutral": 0.45, "happy": 0.05}),
        user_text="Good luck.",
    )
    assert fused["conflict"] is None


def test_not_sad_with_borderline_face_is_not_a_conflict():
    # Log: "I am not sad, I am quite normal, neutral." + face sad 0.4975 was challenged.
    fused = fuse(
        text=text({"neutral": 0.6, "approval": 0.3}),
        face=face({"sad": 0.4975, "neutral": 0.45, "anger": 0.05}),
        user_text="I am not sad, I am quite normal, neutral.",
    )
    assert fused["conflict"] is None


def test_im_fine_with_clearly_sad_face_is_a_conflict():
    fused = fuse(
        text=text({"neutral": 0.8, "approval": 0.2}),
        face=face({"sad": 0.8, "neutral": 0.2}),
        user_text="I'm fine, really.",
    )
    assert fused["conflict"]["observed"] == "sad"
    assert fused["conflict"]["source"] == "face"
    assert fused["conflict"]["minimising_phrase"] is True


def test_neutral_words_with_sad_face_are_not_a_conflict():
    # Log: "Hello, how are you?" / "Wow." with a sad-looking face are not contradictions.
    for utterance in ("Hello, how are you?", "Wow."):
        fused = fuse(text=text({"neutral": 0.9, "curiosity": 0.1}),
                     face=face({"sad": 0.8, "neutral": 0.2}), user_text=utterance)
        assert fused["conflict"] is None, utterance
        assert fused["modalities"]["face"]["emotion"] == "sad"   # still informs the fused read


def test_clearly_positive_words_with_sad_face_are_a_conflict():
    fused = fuse(text=text({"joy": 0.85, "neutral": 0.15}),
                 face=face({"sad": 0.8, "neutral": 0.2}), user_text="Today was great!")
    assert fused["conflict"]["said"] == "positive"


def test_voice_needs_very_high_confidence_to_conflict():
    base = dict(text=text({"neutral": 0.9}), user_text="I'm okay")
    assert fuse(**base, audio=audio({"ang": 0.75, "neu": 0.25}))["conflict"] is None
    assert fuse(**base, audio=audio({"ang": 0.9, "neu": 0.1}))["conflict"]["source"] == "voice"


def test_substrings_do_not_trigger_minimising():
    # "look", "goodbye", "define" contain ok/good/fine but are not minimising.
    for utterance in ("Look at this.", "Goodbye then.", "Can you define it?"):
        fused = fuse(
            text=text({"sadness": 0.7, "neutral": 0.3}),   # words read sad → only a phrase could trigger
            face=face({"sad": 0.9, "neutral": 0.1}),
            user_text=utterance,
        )
        assert fused["conflict"] is None, utterance


def test_no_face_detected_never_conflicts():
    fused = fuse(
        text=text({"neutral": 0.9}),
        face=face({"sad": 0.95}, detected=False),
        user_text="I'm okay",
    )
    assert fused["conflict"] is None
    assert fused["modalities"]["face"]["used"] is False


# ── Weighting / agreement ────────────────────────────────────────────────────

def test_agreeing_modalities_give_confident_read():
    fused = fuse(
        text=text({"joy": 0.9, "neutral": 0.1}),
        audio=audio({"hap": 0.8, "neu": 0.2}),
        face=face({"happy": 0.85, "neutral": 0.15}),
    )
    assert fused["emotion"] == "happy"
    assert fused["agreement"] == "3/3"
    assert fused["confidence"] > 0.75


def test_text_outweighs_a_lone_dissenting_voice_read():
    fused = fuse(
        text=text({"sadness": 0.85, "neutral": 0.15}),
        audio=audio({"ang": 0.6, "neu": 0.4}),
    )
    assert fused["emotion"] == "sad"
    assert fused["agreement"] == "1/2"


def test_gated_modalities_do_not_vote():
    fused = fuse(
        text=text({"neutral": 0.9, "joy": 0.1}),
        audio=audio({"ang": 0.33, "neu": 0.32, "hap": 0.2, "sad": 0.15}, gated=True),
    )
    assert fused["modalities"]["audio"]["used"] is False
    assert fused["emotion"] == "neutral"


def test_nothing_available_is_neutral_with_zero_confidence():
    fused = fuse()
    assert fused["emotion"] == "neutral"
    assert fused["confidence"] == 0.0
    assert fused["agreement"] == "0/0"


def test_dsmr_valence_is_only_a_gentle_prior():
    # Near-tie: the sentiment prior may tip it.
    near_tie = dict(text=text({"neutral": 0.55, "sadness": 0.45}))
    assert fuse(**near_tie)["emotion"] == "neutral"
    assert fuse(**near_tie, dsmr_valence=-0.9)["emotion"] == "sad"
    # Clear read: the prior nudges but does not override the words.
    clear = dict(text=text({"neutral": 0.7, "sadness": 0.3}))
    negative = fuse(**clear, dsmr_valence=-0.9)
    assert negative["distribution"]["sad"] > fuse(**clear)["distribution"]["sad"]
    assert negative["emotion"] == "neutral"


def test_goemotions_aggregate_by_category():
    core = to_core({"gratitude": 0.5, "joy": 0.2, "neutral": 0.3}, _GOEMOTIONS_TO_CORE)
    assert core["happy"] == pytest.approx(0.7)
    assert core["neutral"] == pytest.approx(0.3)


# ── Face baseline calibration ────────────────────────────────────────────────

def test_resting_sad_face_is_calibrated_toward_neutral():
    baseline = {"neutral": 0.4, "sad": 0.5, "happy": 0.05, "angry": 0.05, "fearful": 0.0, "surprised": 0.0}
    dist = {"neutral": 0.42, "sad": 0.52, "happy": 0.06, "angry": 0.0, "fearful": 0.0, "surprised": 0.0}
    calibrated = calibrate_face(dist, baseline, observations=5)
    assert calibrated["neutral"] > calibrated["sad"]
    # Not applied before enough observations.
    assert calibrate_face(dist, baseline, observations=1) is dist


def test_session_baseline_flows_through_store():
    store = ConversationStore()
    state = store.get("s1")
    resting = {"sad": 0.55, "neutral": 0.4, "happy": 0.05}
    for _ in range(4):
        fused = fuse(text=text({"neutral": 0.9}), face=face(resting),
                     face_baseline=state.face_baseline, face_observations=state.face_observations)
        state.update_face_baseline(fused.pop("face_raw_distribution"))
    fused = fuse(text=text({"neutral": 0.9}), face=face(resting),
                 face_baseline=state.face_baseline, face_observations=state.face_observations)
    assert fused["modalities"]["face"]["emotion"] == "neutral"
    assert "calibrated" in fused["modalities"]["face"]["note"]


# ── DMESR sentiment / session state ──────────────────────────────────────────

def test_dsmr_is_reported_as_sentiment():
    assert sentiment_band(-0.6) == "negative"
    assert sentiment_band(-0.2) == "slightly negative"
    assert sentiment_band(0.0) == "neutral"
    assert sentiment_band(0.7) == "positive"
    d = dsmr_sentiment({"valence": -0.24})
    assert d["emotion"] == "slightly negative" and d["valence"] == -0.24


def test_trajectory_and_conflict_cooldown():
    state = ConversationStore().get("s2")
    state.add_user_turn("hi", "happy", 0.8)
    state.add_user_turn("meh", "sad", 0.7)
    assert state.trajectory() == "shifted from happy to sad"
    state.add_user_turn("still bad", "sad", 0.6)
    assert state.trajectory() == "sad for the last 2 turns"

    assert state.may_raise_conflict()
    state.mark_conflict_raised()
    assert not state.may_raise_conflict()
    for _ in range(4):
        state.add_user_turn("x", "neutral", 0.5)
    assert state.may_raise_conflict()


def test_store_lru_and_history():
    store = ConversationStore(max_sessions=2, max_turns=3)
    a = store.get("a")
    for i in range(5):
        a.add_user_turn(f"u{i}", "neutral", 0.5)
    assert [m["content"] for m in a.history_messages()] == ["u2", "u3", "u4"]
    store.get("b")
    store.get("c")
    assert len(store) == 2 and "a" not in store._sessions


def test_history_is_capped_clipped_and_carries_emotion_reads():
    state = ConversationStore().get("h")
    state.add_assistant_turn("orphan reply", "neutral")
    for i in range(5):
        state.add_user_turn(f"u{i} " + "x" * 600, "sad", 0.7, {"face": ("sad", 0.8)})
        state.add_assistant_turn(f"a{i}", "empathetic")
    history = state.history_messages()
    assert len(history) == HISTORY_MESSAGES
    assert history[0]["role"] == "user" and history[0]["content"].startswith("u2 ")
    assert len(history[0]["content"]) == HISTORY_MESSAGE_MAX_CHARS
    assert history[0]["signals"] == {"face": ("sad", 0.8)} and history[0]["emotion"] == "sad"
    assert "signals" not in history[1]


def test_history_never_opens_with_an_assistant_message():
    state = ConversationStore().get("o")
    state.add_user_turn("u0", "neutral", 0.5)
    state.add_assistant_turn("a0", "neutral")
    state.add_user_turn("u1", "neutral", 0.5)           # LLM failed: no assistant turn stored
    state.add_assistant_turn("a1", "neutral")
    assert [m["content"] for m in state.history_messages(last_n=3)] == ["u1", "a1"]


def test_nonverbal_reads_skip_missing_signals():
    fused = fuse(text={"available": True, "probabilities": {"neutral": 1.0}},
                 face={"available": True, "face_detected": False})
    assert nonverbal_reads(fused) == {}
    fused = fuse(audio={"available": True, "probabilities": {"ang": 0.9, "neu": 0.1}})
    assert nonverbal_reads(fused) == {"voice": ("angry", 0.9)}
