import pytest

from llm_service import (
    _build_messages,
    _build_system_prompt,
    _build_visual_chat_prompt,
    _parse_emotion_sentence,
    _parse_llm_response,
    asks_about_own_emotion,
    response_voice_params,
)
from emotion_fusion import fuse


# ── emotion(sentence) ────────────────────────────────────────────────────────

def test_clean_emotion_sentence():
    assert _parse_emotion_sentence("happy(That's great news!)") == ("happy", "That's great news!")


def test_missing_closing_paren_from_research_log():
    # Real reply that leaked to TTS/UI as "concerned(I'm really sorry…".
    raw = "concerned(I'm really sorry you're frustrated, I'm here and listening."
    assert _parse_emotion_sentence(raw) == (
        "concerned", "I'm really sorry you're frustrated, I'm here and listening."
    )


def test_think_block_is_stripped():
    raw = "<think>The user seems sad, I should validate.</think>\nempathetic(That sounds really hard.)"
    assert _parse_emotion_sentence(raw) == ("empathetic", "That sounds really hard.")


def test_unterminated_think_block_falls_back():
    emotion, sentence = _parse_emotion_sentence("<think>Let me reason about this for a while")
    assert emotion == "neutral"
    assert "<think>" not in sentence


def test_trailing_explanation_is_dropped():
    raw = "supportive(I'm right here with you.)\n\nThis reply validates the user's feelings."
    assert _parse_emotion_sentence(raw) == ("supportive", "I'm right here with you.")


def test_inner_parentheses_kept():
    raw = "happy(That's (honestly) amazing!)"
    assert _parse_emotion_sentence(raw) == ("happy", "That's (honestly) amazing!")


def test_listening_maps_to_neutral_and_synonyms_map():
    assert _parse_emotion_sentence("listening(Take your time.)")[0] == "neutral"
    assert _parse_emotion_sentence("joyful(Yay!)")[0] == "happy"
    assert _parse_emotion_sentence("caring(I'm here.)")[0] == "empathetic"


def test_label_then_sentence_from_research_log():
    raw = "emotion(listening) I'm doing well, thanks for asking! How are you feeling today?"
    assert _parse_emotion_sentence(raw) == (
        "neutral", "I'm doing well, thanks for asking! How are you feeling today?"
    )


def test_punctuation_after_closing_paren_from_research_log():
    raw = "concerned(Sounds like your mind has been really loud lately. I'm here if you want to vent)."
    assert _parse_emotion_sentence(raw) == (
        "concerned", "Sounds like your mind has been really loud lately. I'm here if you want to vent."
    )


def test_emoji_removed_for_tts():
    assert _parse_emotion_sentence("happy(You're very welcome! 😊)") == ("happy", "You're very welcome!")


def test_colon_style_reply():
    assert _parse_emotion_sentence("happy: Good to see you!") == ("happy", "Good to see you!")


def test_plain_text_fallback():
    assert _parse_emotion_sentence("Hello there!") == ("neutral", "Hello there!")


# ── JSON segments ────────────────────────────────────────────────────────────

def test_json_array():
    raw = '[{"text": "I hear you.", "emotion": "compassion"}]'
    assert _parse_llm_response(raw) == [{"text": "I hear you.", "emotion": "compassion"}]


def test_json_with_prose_and_fences():
    raw = 'Sure! Here you go:\n```json\n[{"text": "Hi!", "emotion": "happy"}]\n```'
    assert _parse_llm_response(raw) == [{"text": "Hi!", "emotion": "happy"}]


def test_json_object_with_segments_and_unknown_emotion():
    raw = '{"segments": [{"text": "Okay.", "emotion": "joyful"}, {"text": "Hm.", "emotion": "weird"}]}'
    assert _parse_llm_response(raw) == [
        {"text": "Okay.", "emotion": "happy"},
        {"text": "Hm.", "emotion": "neutral"},
    ]


def test_json_garbage_falls_back_to_text():
    assert _parse_llm_response("just words") == [{"text": "just words", "emotion": "neutral"}]


# ── Prompt / messages ────────────────────────────────────────────────────────

def test_messages_include_history_in_order():
    history = [
        {"role": "user", "content": "I lost my job today."},
        {"role": "assistant", "content": "I'm so sorry."},
    ]
    msgs = _build_messages("SYS", history, "because of you.")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"] == "because of you."


def test_visual_prompt_mentions_conflict_only_when_allowed():
    fused = fuse(
        text={"available": True, "probabilities": {"neutral": 0.9, "approval": 0.1}},
        face={"available": True, "face_detected": True, "probabilities": {"sad": 0.85, "neutral": 0.15}},
        user_text="I'm fine",
    )
    assert fused["conflict"]
    allowed = _build_visual_chat_prompt(fused, None, "", allow_conflict_mention=True)
    blocked = _build_visual_chat_prompt(fused, None, "", allow_conflict_mention=False)
    assert "MISMATCH NOTICED" in allowed
    assert "Do not bring it up again" in blocked
    assert "listening" not in allowed.split("one of:")[-1].split("\n")[0]


@pytest.mark.parametrize("text", [
    "can i u say wt emotion i am showing now",      # real (garbled) request that went unanswered
    "What emotion am I showing?",
    "do I look sad",
    "am I happy or sad?",
    "guess my mood",
    "Can you tell what I'm feeling?",
    "which emotion is on my face",
])
def test_detects_questions_about_own_emotion(text):
    assert asks_about_own_emotion(text)


@pytest.mark.parametrize("text", [
    "My mood is terrible today",
    "How do I tell my boss about my feelings?",
    "My feelings are hurt because of you",
    "how are you?",
    "I'm fine",
])
def test_ordinary_messages_are_not_emotion_questions(text):
    assert not asks_about_own_emotion(text)


def _visual_fused(question):
    return fuse(
        text={"available": True, "probabilities": {"neutral": 0.95, "curiosity": 0.05}},
        audio={"available": True, "probabilities": {"neu": 0.85, "sad": 0.15}},
        face={"available": True, "face_detected": True, "probabilities": {"happy": 0.72, "neutral": 0.28}},
        user_text=question,
    )


def test_emotion_question_points_the_model_at_face_and_voice():
    question = "what emotion am I showing now?"
    fused = _visual_fused(question)
    assert fused["emotion"] == "neutral"        # the question's words dominate the overall read
    prompt = _build_visual_chat_prompt(fused, None, "", user_text=question)
    this_turn = prompt.split("THIS TURN:")[1].split("\n")[0]
    assert "face happy 72%" in this_turn and "voice neutral 85%" in this_turn
    assert "THIS TURN:" not in _build_visual_chat_prompt(fused, None, "", user_text="I had a long day")


def test_emotion_question_in_text_chat_admits_words_only():
    question = "can you tell what I'm feeling?"
    fused = fuse(text={"available": True, "probabilities": {"neutral": 0.9, "curiosity": 0.1}}, user_text=question)
    prompt = _build_system_prompt(fused, None, "", user_text=question)
    assert "only go by their words" in prompt
    assert "can't see their face" in prompt


def test_prompt_lists_emotion_history_of_earlier_user_messages():
    history = [
        {"role": "user", "content": "I lost my job today", "emotion": "sad", "confidence": 0.72,
         "signals": {"face": ("sad", 0.8), "voice": ("neutral", 0.9)}},
        {"role": "assistant", "content": "I'm so sorry."},
        {"role": "user", "content": "whatever", "emotion": "neutral", "confidence": 0.0, "signals": {}},
    ]
    prompt = _build_visual_chat_prompt(_visual_fused("hi"), None, "", history=history, user_text="hi")
    block = prompt.split("EMOTION HISTORY")[1].split("\n\n")[0]
    assert '"I lost my job today" → sad 72% (face sad 80%, voice neutral 90%)' in block
    assert "whatever" not in block                 # no usable read for that turn
    assert "I'm so sorry" not in block             # assistant turns carry no user read


def test_response_voice_params_follow_reply_emotion():
    assert response_voice_params("sad")["speed"] < 1.0
    assert response_voice_params("happy")["speed"] > 1.0
    assert response_voice_params("unknown")["emotion"] == "unknown"
