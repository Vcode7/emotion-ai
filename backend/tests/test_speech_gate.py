import math

import torch

from speech_gate import analyse_speech, is_no_speech, normalise_transcript, trim_silence

SR = 16_000


def tone(seconds, amp=0.3, freq=220.0):
    t = torch.arange(int(seconds * SR)) / SR
    return amp * torch.sin(2 * math.pi * freq * t)


def silence(seconds, noise=0.0005):
    return noise * torch.randn(int(seconds * SR))


def test_silence_has_no_voice():
    stats = analyse_speech(silence(3.0), SR)
    assert stats.voiced_sec == 0.0
    assert is_no_speech("Thank you.", [], stats.voiced_sec)[0]


def test_click_is_too_short():
    wave = torch.cat([silence(1.0), tone(0.08, amp=0.5), silence(1.0)])
    stats = analyse_speech(wave, SR)
    assert stats.voiced_sec < 0.3
    no_speech, reason = is_no_speech("Thank you.", [], stats.voiced_sec)
    assert no_speech and "too_little_voice" in reason


def test_real_speech_passes_even_with_thank_you_words():
    wave = torch.cat([silence(1.0), tone(2.0), silence(0.9)])
    stats = analyse_speech(wave, SR)
    assert 1.8 <= stats.voiced_sec <= 2.1
    assert is_no_speech("Thank you so much for helping me today", [], stats.voiced_sec) == (False, "")


def test_hallucination_phrase_with_weak_voice_is_dropped():
    wave = torch.cat([silence(2.0), tone(0.5, amp=0.05), silence(1.0)])
    stats = analyse_speech(wave, SR)
    assert is_no_speech("Thank you.", [], stats.voiced_sec)[0]


def test_hallucination_phrase_with_clear_voice_is_kept():
    stats = analyse_speech(torch.cat([silence(0.5), tone(1.2), silence(0.5)]), SR)
    segments = [{"no_speech_prob": 0.02, "avg_logprob": -0.2}]
    assert is_no_speech("Thank you.", segments, stats.voiced_sec) == (False, "")


def test_whisper_no_speech_rule():
    segments = [{"no_speech_prob": 0.9, "avg_logprob": -1.4}]
    assert is_no_speech("Some words", segments, 1.5) == (True, "whisper_no_speech")


def test_empty_and_inaudible():
    assert is_no_speech("", [], None)[0]
    assert is_no_speech("(inaudible)", [], None)[0]
    assert is_no_speech("  ...  ", [], None)[0]


def test_unknown_voicing_does_not_block_normal_text():
    assert is_no_speech("I had a rough day", [], None) == (False, "")


def test_trim_silence_keeps_speech_with_padding():
    wave = torch.cat([silence(5.0), tone(1.0), silence(2.0)])
    trimmed = trim_silence(wave, SR)
    assert 1.0 * SR <= trimmed.numel() <= 1.5 * SR


def test_normalise_transcript():
    assert normalise_transcript("  Thank you!! ") == "thank you"
    assert normalise_transcript("I'm FINE.") == "im fine"
