"""
llm_service.py — Multi-provider, emotion-aware LLM response generation
=======================================================================

Builds an emotion-aware system prompt from the fused multimodal emotion
read (see ``emotion_fusion.py``) plus recent conversation history, and
routes the request to one of three providers:

    • **OpenAI**  — ``gpt-4o-mini`` via the official async client
    • **Groq**    — ``llama-3.3-70b-versatile`` via OpenAI-compatible endpoint
    • **Gemini**  — ``gemini-2.0-flash`` via ``google-genai`` (async)

Two response formats:

    generate_response()              → JSON array of segments
        [{"text": "…", "emotion": "compassion"}, …]
    generate_visual_chat_response()  → one line ``emotion(sentence)``
        (Visual Chat; defaults to Qwen 27B via Groq)

Every provider receives ``[system, *history, user]`` messages so the model
remembers the conversation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Valid emotion labels the LLM may use in its response segments.
VALID_EMOTIONS = frozenset(
    ["compassion", "supportive", "encouraging", "concerned",
     "neutral", "happy", "thinking", "sad"]
)

# ── Visual Chat emotion vocabulary ────────────────────────────────────────────
# Every label here has an avatar preset on the frontend (emotionPresets.js).
VALID_VISUAL_EMOTIONS = frozenset([
    "neutral", "happy", "sad", "angry", "surprised",
    "concerned", "empathetic", "thinking",
    "compassion", "supportive", "encouraging", "fearful",
])

# The companion should never answer angrily or fearfully, so the prompt only
# offers these (the parser still accepts the full set above).
_PROMPT_VISUAL_EMOTIONS = (
    "neutral", "happy", "empathetic", "compassion", "supportive",
    "encouraging", "concerned", "thinking", "surprised", "sad",
)

# Map any LLM emotion to the closest supported avatar state.
_AVATAR_FALLBACK_MAP: Dict[str, str] = {
    "joy":          "happy",
    "joyful":       "happy",
    "happiness":    "happy",
    "excitement":   "happy",
    "excited":      "happy",
    "cheerful":     "happy",
    "warm":         "happy",
    "grief":        "sad",
    "sadness":      "sad",
    "sorrow":       "sad",
    "fear":         "fearful",
    "anxiety":      "concerned",
    "worry":        "concerned",
    "worried":      "concerned",
    "frustration":  "angry",
    "disgust":      "angry",
    "contempt":     "angry",
    "shock":        "surprised",
    "awe":          "surprised",
    "calm":         "neutral",
    "serious":      "neutral",
    "listening":    "neutral",
    "speaking":     "neutral",
    "caring":       "empathetic",
    "compassionate": "compassion",
    "empathy":      "empathetic",
    "reassuring":   "supportive",
    "confused":     "thinking",
    "curious":      "thinking",
    "pondering":    "thinking",
}

# Voice delivery for each AI response emotion (used by TTS + Web Speech).
_RESPONSE_VOICE: Dict[str, Tuple[float, float, float]] = {
    # emotion:      (speed, pitch, warmth)
    "happy":        (1.05, 1.05, 0.8),
    "encouraging":  (1.04, 1.04, 0.8),
    "surprised":    (1.05, 1.08, 0.7),
    "sad":          (0.92, 0.97, 0.9),
    "compassion":   (0.92, 0.98, 1.0),
    "empathetic":   (0.94, 0.98, 1.0),
    "concerned":    (0.95, 0.98, 0.9),
    "supportive":   (0.97, 1.0, 0.9),
    "thinking":     (0.97, 1.0, 0.7),
    "neutral":      (1.0, 1.0, 0.7),
}


def normalise_response_emotion(label: Optional[str], valid=VALID_VISUAL_EMOTIONS) -> str:
    emotion = str(label or "neutral").strip().lower()
    if emotion in valid:
        return emotion
    mapped = _AVATAR_FALLBACK_MAP.get(emotion)
    if mapped and mapped in valid:
        return mapped
    return "neutral"


def response_voice_params(emotion: str) -> Dict[str, Any]:
    """Voice parameters that match the AI's own response emotion."""
    speed, pitch, warmth = _RESPONSE_VOICE.get(emotion, _RESPONSE_VOICE["neutral"])
    return {
        "speed": speed,
        "pitch": pitch,
        "warmth": warmth,
        "emotion": emotion,
        "pause_after_sentence_ms": 350 if emotion in {"sad", "compassion", "empathetic", "concerned"} else 200,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

_STRATEGY = {
    "sad":       "Validate the feeling first, then offer one gentle, open question. Don't rush to fix it.",
    "angry":     "Acknowledge the frustration without getting defensive, take it seriously, and offer something concrete.",
    "fearful":   "Be calm and grounding; reassure without dismissing the worry, and keep it simple.",
    "happy":     "Share their energy and be curious about what's going well.",
    "surprised": "Match their interest and help them make sense of what surprised them.",
    "neutral":   "Have a natural, friendly conversation; don't invent feelings they haven't shown.",
}

_CONFIDENCE_RULES = (
    "- Overall-read confidence below 55%: do not name an emotion; just respond warmly to their words.\n"
    "- 55–75%: you may hedge (\"you sound a bit…\", \"it seems like…\").\n"
    "- Above 75%: you may name the feeling directly, but never as a certainty.\n"
    "- Only refer to their face (\"you look…\") when the face read is at least 70%."
)


def _pct(x: Any) -> int:
    try:
        return round(float(x) * 100)
    except (TypeError, ValueError):
        return 0


def _format_modality(name: str, read: Dict[str, Any], nuance: List[str]) -> str:
    if not read or not read.get("available"):
        return f"- {name}: {read.get('note') or 'unavailable'}" if read else f"- {name}: unavailable"
    line = f"- {name}: {read.get('emotion')} {_pct(read.get('confidence'))}%"
    extras = []
    if nuance:
        extras.append("nuance: " + ", ".join(nuance))
    if read.get("note"):
        extras.append(read["note"])
    if not read.get("used"):
        extras.append("not counted")
    if extras:
        line += f" ({'; '.join(extras)})"
    return line


def _perception_block(
    fused: Dict[str, Any],
    dsmr: Optional[Dict[str, Any]],
    trajectory: str,
    include_face: bool,
) -> str:
    mods = fused.get("modalities") or {}
    lines = [
        "PERCEPTION OF THE USER (automated estimates — they can be wrong):",
        f"- Overall read: {fused.get('emotion', 'neutral')} "
        f"{_pct(fused.get('confidence'))}% (signals agreeing: {fused.get('agreement', '0/0')})",
        _format_modality("Words", mods.get("text", {}), fused.get("nuance") or []),
        _format_modality("Voice tone", mods.get("audio", {}), []),
    ]
    if include_face:
        lines.append(_format_modality("Face", mods.get("face", {}), []))
    if dsmr:
        lines.append(
            f"- Sentiment (DMESR): {dsmr.get('emotion', 'neutral')} "
            f"(valence {float(dsmr.get('valence', 0.0)):+.2f})"
        )
    if trajectory:
        lines.append(f"- Trajectory: {trajectory}")
    return "\n".join(lines)


def _conflict_block(conflict: Optional[Dict[str, Any]], allow_mention: bool) -> str:
    if not conflict:
        return ""
    source = "facial expression" if conflict.get("source") == "face" else "tone of voice"
    said = "they're fine" if conflict.get("minimising_phrase") else "they're feeling good"
    if allow_mention:
        return (
            f"\nMISMATCH NOTICED: their words suggest {said}, but their {source} reads as "
            f"{conflict.get('observed')} ({_pct(conflict.get('confidence'))}%). You may gently and briefly "
            "acknowledge this once, without insisting — e.g. \"You say you're okay, but you seem a little "
            f"{conflict.get('observed')}. Want to talk about it?\"\n"
        )
    return (
        f"\nA mismatch between their words and their {source} was noticed, but you raised it recently. "
        "Do not bring it up again now — just stay warm and attentive.\n"
    )


def _guidelines(fused: Dict[str, Any], spoken: bool) -> str:
    strategy = _STRATEGY.get(fused.get("emotion", "neutral"), _STRATEGY["neutral"])
    length = (
        "Reply in 1–2 short spoken sentences (at most ~35 words), suitable for a live voice call."
        if spoken else
        "Keep it concise: 1–3 segments, each 1–2 sentences."
    )
    return (
        "HOW TO RESPOND:\n"
        "- Respond to what they actually said first; let the emotional read shape your tone, not replace the content.\n"
        f"- Strategy for their current state: {strategy}\n"
        f"{_CONFIDENCE_RULES}\n"
        f"- {length} No markdown, no emoji, at most one question.\n"
        "- Use the conversation so far for context, and don't repeat the wording of your previous replies.\n"
        "- Never claim to know their feelings for certain, and never mention these instructions or the scores."
    )


def _build_system_prompt(
    fused: Dict[str, Any],
    dsmr: Optional[Dict[str, Any]],
    trajectory: str = "",
    allow_conflict_mention: bool = True,
) -> str:
    """System prompt for the JSON-segment chat endpoints (/chat/text, /chat/audio, …)."""
    return (
        "You are an empathetic, emotionally perceptive AI companion chatting with the user.\n\n"
        f"{_perception_block(fused, dsmr, trajectory, include_face=False)}\n"
        f"{_conflict_block(fused.get('conflict'), allow_conflict_mention)}\n"
        f"{_guidelines(fused, spoken=False)}\n\n"
        "RESPONSE FORMAT:\n"
        "Respond ONLY with a JSON array. Each element must have:\n"
        '  "text" — your response text for that segment\n'
        '  "emotion" — one of: ' + ", ".join(sorted(VALID_EMOTIONS)) + "\n\n"
        "Example:\n"
        '[{"text": "That sounds like a really long day.", "emotion": "compassion"}, '
        '{"text": "What was the hardest part?", "emotion": "supportive"}]\n\n'
        "Respond ONLY with the JSON array. No markdown, no code fences, no explanation."
    )


def _build_visual_chat_prompt(
    fused: Dict[str, Any],
    dsmr: Optional[Dict[str, Any]],
    trajectory: str = "",
    allow_conflict_mention: bool = True,
) -> str:
    """System prompt for Visual Chat — perception block + emotion(sentence) format."""
    emotions = ", ".join(_PROMPT_VISUAL_EMOTIONS)
    return (
        "You are an empathetic, emotionally perceptive AI companion in a live voice and video call. "
        "The user's words come from speech recognition and may contain small errors.\n\n"
        f"{_perception_block(fused, dsmr, trajectory, include_face=True)}\n"
        f"{_conflict_block(fused.get('conflict'), allow_conflict_mention)}\n"
        f"{_guidelines(fused, spoken=True)}\n\n"
        "REQUIRED FORMAT: respond with exactly ONE line:\n"
        "emotion(Your response sentence here)\n"
        f"where emotion is your own reaction, one of: {emotions}.\n\n"
        "Examples:\n"
        "empathetic(That sounds exhausting. What's been weighing on you the most?)\n"
        "happy(That's wonderful news! How are you going to celebrate?)\n"
        "concerned(You say you're okay, but you seem a little down. Want to talk about it?)\n"
        "neutral(Sure, I can help with that. What would you like to start with?)"
    )


def _build_messages(system_prompt: str, history: Optional[List[Dict[str, str]]],
                    user_text: str) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    for m in history or []:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_text})
    return messages


# ─────────────────────────────────────────────────────────────────────────────
#  Response parsing
# ─────────────────────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _clean_raw(raw: str) -> str:
    """Remove reasoning blocks and markdown fences from a model reply."""
    text = _THINK_RE.sub("", raw or "")
    lower = text.lower()
    if "<think>" in lower:            # unterminated reasoning — the answer never came
        text = text[: lower.index("<think>")]
    text = text.strip()
    text = re.sub(r"^```[\w]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_llm_response(raw: str) -> List[Dict[str, str]]:
    """
    Parse the raw LLM output into a list of ``{text, emotion}`` dicts.

    Handles:
      - Clean JSON arrays
      - JSON wrapped in markdown code fences or surrounded by prose
      - ``{"segments": [...]}`` style objects
      - Completely malformed output (falls back to neutral)
    """
    cleaned = _clean_raw(raw)

    candidates = [cleaned]
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match and match.group(0) != cleaned:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            parsed = parsed.get("segments") or parsed.get("response") or []
        if not isinstance(parsed, list):
            continue
        result: List[Dict[str, str]] = []
        for item in parsed:
            text = _tidy_sentence(str(item.get("text", "")), strip_quotes=False) if isinstance(item, dict) else ""
            if text:
                emotion = normalise_response_emotion(item.get("emotion"), VALID_EMOTIONS)
                result.append({"text": text, "emotion": emotion})
        if result:
            return result

    # Fallback: treat the entire response as a single neutral segment
    logger.warning("LLM response was not valid JSON — using raw text fallback.")
    return [{"text": cleaned or "I'm here to help.", "emotion": "neutral"}]


_EMOTION_CALL_RE = re.compile(r"^[\s\"'*`>-]*([A-Za-z_]+)\s*\(", re.MULTILINE)
# ")" ending the sentence, optionally followed by punctuation the model put outside it.
_CLOSING_PAREN_RE = re.compile(r"\)\s*([.!?]*)[\"'*`]*\s*(?:\n|$)")
# "emotion(listening) I'm doing well" — a lone label in parentheses, then the sentence.
_LABEL_THEN_SENTENCE_RE = re.compile(r"^\s*([A-Za-z_]+)\s*\)\s*(\S.*)$", re.DOTALL)
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF️‍]+")


def _tidy_sentence(text: str, strip_quotes: bool = True) -> str:
    """Collapse whitespace, drop emoji (TTS would read them out) and stray quotes."""
    text = re.sub(r"\s+", " ", _EMOJI_RE.sub("", text)).strip()
    return text.strip("\"'").strip() if strip_quotes else text


def _parse_emotion_sentence(raw: str) -> Tuple[str, str]:
    """
    Parse the LLM's ``emotion(sentence)`` response.

    Handles:
        - ``concerned(I understand.)`` / ``happy (Things are great!)``
        - a missing closing parenthesis or trailing explanation lines
        - ``<think>…</think>`` reasoning blocks and markdown fences
        - ``happy: sentence`` style replies
        - Completely malformed output (falls back to neutral + text)

    Returns
    -------
    (emotion, sentence) both as strings.
    """
    cleaned = _clean_raw(raw)

    known = VALID_VISUAL_EMOTIONS | set(_AVATAR_FALLBACK_MAP)

    m = _EMOTION_CALL_RE.search(cleaned)
    if m:
        raw_emotion = m.group(1).lower()
        rest = cleaned[m.end():]

        single = _LABEL_THEN_SENTENCE_RE.match(rest.split("\n", 1)[0])
        if single and (raw_emotion == "emotion" or single.group(1).lower() in known):
            raw_emotion, sentence = single.group(1).lower(), _tidy_sentence(single.group(2))
        else:
            close = _CLOSING_PAREN_RE.search(rest)
            if close:
                sentence = _tidy_sentence(rest[: close.start()])
                trailing = close.group(1)
                if trailing and sentence and sentence[-1] not in ".!?":
                    sentence += trailing[0]
            else:
                sentence = _tidy_sentence(rest.split("\n", 1)[0])

        if sentence:
            emotion = normalise_response_emotion(raw_emotion)
            if emotion == "neutral" and raw_emotion not in ("neutral", "listening", "speaking", "calm", "serious"):
                logger.warning("[ParseEmotionSentence] Unknown emotion '%s' — defaulting to neutral.", raw_emotion)
            return emotion, sentence

    # "happy: sentence" / "happy - sentence"
    m = re.match(r"^\s*([A-Za-z_]+)\s*[:\-–]\s*(.+)$", cleaned, re.DOTALL)
    if m and m.group(1).lower() in known:
        sentence = _tidy_sentence(m.group(2).split("\n", 1)[0])
        if sentence:
            return normalise_response_emotion(m.group(1)), sentence

    logger.warning(
        "[ParseEmotionSentence] Could not parse emotion(sentence) from: %r — using neutral fallback.",
        (raw or "")[:120],
    )
    # Keep the content, but only the first paragraph — never speak explanations.
    sentence = _tidy_sentence(cleaned.split("\n\n", 1)[0])
    return "neutral", sentence or "I'm here with you."


# ─────────────────────────────────────────────────────────────────────────────
#  Provider-specific callers
# ─────────────────────────────────────────────────────────────────────────────

async def _call_openai(
    messages: List[Dict[str, str]],
    api_key: str,
    model: str,
    max_tokens: int = 250,
) -> str:
    """Call the OpenAI Chat Completions API."""
    import openai

    client = openai.AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


async def _call_groq(
    messages: List[Dict[str, str]],
    api_key: str,
    model: str,
    max_tokens: int = 250,
) -> str:
    """Call Groq via the OpenAI-compatible endpoint with automatic fallback on rate limit."""
    import openai
    from config import get_groq_api_keys

    keys_to_try = get_groq_api_keys(api_key)
    if not keys_to_try:
        raise ValueError("No Groq API keys available (checked primary key and fallback keys).")

    last_exc = None
    for idx, key in enumerate(keys_to_try):
        try:
            client = openai.AsyncOpenAI(
                api_key=key,
                base_url="https://api.groq.com/openai/v1",
            )
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=max_tokens,
            )
            if idx > 0:
                logger.info("[Groq LLM] Fallback key #%d succeeded (model: %s).", idx, model)
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            is_rate_limit = (
                isinstance(exc, openai.RateLimitError)
                or getattr(exc, "status_code", None) == 429
                or any(phrase in str(exc).lower() for phrase in [
                    "rate_limit", "rate limit", "429", "tokens per minute", "requests per minute", "otpm", "tpm", "rpm"
                ])
            )
            if is_rate_limit:
                has_next = idx + 1 < len(keys_to_try)
                if has_next:
                    logger.warning(
                        "[Groq LLM] Rate limit hit on key %s... (error: %s). Switching to fallback key #%d...",
                        key[:10], str(exc)[:120], idx + 1
                    )
                    continue
                else:
                    logger.error(
                        "[Groq LLM] Rate limit hit on key %s... and all %d Groq fallback keys were exhausted.",
                        key[:10], len(keys_to_try)
                    )
            raise last_exc

    if last_exc:
        raise last_exc
    return ""


async def _call_gemini(
    messages: List[Dict[str, str]],
    api_key: str,
    model: str,
    max_tokens: int = 250,
) -> str:
    """Call Google Gemini via the async google-genai client."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part(text=m["content"])],
        )
        for m in messages
        if m["role"] != "system"
    ]
    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.7,
        ),
    )
    return response.text or ""


_CALLERS = {"openai": _call_openai, "groq": _call_groq, "gemini": _call_gemini}

# Default models per provider
_DEFAULT_MODELS = {
    "openai":   "gpt-4o-mini",
    "groq":     "llama-3.3-70b-versatile",
    "gemini":   "gemini-2.0-flash",
    # Qwen 27B via Groq
    "qwen27b":  "qwen/qwen3.8-27b",
}

# Reasoning-capable models spend tokens thinking before they answer.
_VISUAL_MAX_TOKENS = 300


def _resolve_api_key(provider: str, api_key: Optional[str]) -> str:
    if api_key:
        return api_key
    from config import OPENAI_API_KEY, GEMINI_API_KEY, get_groq_api_keys  # noqa: PLC0415
    if provider == "groq":
        groq_keys = get_groq_api_keys()
        return groq_keys[0] if groq_keys else ""
    if provider == "openai":
        return OPENAI_API_KEY
    if provider == "gemini":
        return GEMINI_API_KEY
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Public Service Class
# ─────────────────────────────────────────────────────────────────────────────

class LLMService:
    """
    Stateless multi-provider LLM service.

    Builds an emotion-aware system prompt from the fused emotion read,
    adds the conversation history, routes the request to the selected
    provider, and parses the response.
    """

    async def generate_response(
        self,
        user_text: str,
        fused: Dict[str, Any],
        dsmr: Optional[Dict[str, Any]],
        provider: str,
        api_key: Optional[str],
        model: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        trajectory: str = "",
        allow_conflict_mention: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Generate an emotion-aware response as JSON segments.

        Args:
            user_text:   The user's input text.
            fused:       Output of ``emotion_fusion.fuse()``.
            dsmr:        DMESR sentiment dict (``emotion_fusion.dsmr_sentiment()``).
            provider:    ``"openai"``, ``"groq"``, or ``"gemini"``.
            api_key:     Provider API key (optional, falls back to env/config).
            model:       Optional model override.
            history:     Previous turns as ``[{"role", "content"}]``, oldest first.
            trajectory:  Short description of the user's recent emotional trajectory.
            allow_conflict_mention: Whether a words-vs-nonverbal mismatch may be raised.

        Returns:
            List of ``{"text": str, "emotion": str}`` segments.
        """
        effective_model = model or _DEFAULT_MODELS.get(provider, "gpt-4o-mini")
        effective_key = _resolve_api_key(provider, api_key)

        if not effective_key:
            logger.error("No API key provided for provider: %s", provider)
            return [{
                "text": f"LLM error: No API key provided for '{provider}'. Please configure it in settings or the .env file.",
                "emotion": "neutral"
            }]

        caller = _CALLERS.get(provider)
        if caller is None:
            logger.error("Unknown LLM provider: %s", provider)
            return [{"text": f"Unknown LLM provider: {provider}", "emotion": "neutral"}]

        system_prompt = _build_system_prompt(fused, dsmr, trajectory, allow_conflict_mention)
        messages = _build_messages(system_prompt, history, user_text)

        try:
            raw = await caller(messages, effective_key, effective_model)
        except Exception as exc:
            logger.exception("LLM API call failed (%s / %s).", provider, effective_model)
            return [{"text": f"LLM error: {exc}", "emotion": "neutral"}]

        return _parse_llm_response(raw)

    # ──────────────────────────────────────────────────────────────────
    # Visual Chat: fused multimodal emotion → emotion(sentence)
    # ──────────────────────────────────────────────────────────────────

    async def generate_visual_chat_response(
        self,
        user_text: str,
        fused: Dict[str, Any],
        dsmr: Optional[Dict[str, Any]],
        provider: str = "groq",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        trajectory: str = "",
        allow_conflict_mention: bool = True,
    ) -> Tuple[str, str]:
        """
        Generate an emotion-aware Visual Chat response.

        Defaults to Qwen 27B via Groq, falling back to other Groq models if
        the primary model ID is rejected.

        Returns
        -------
        (emotion, sentence) — both strings, safe to use directly.
        """
        # ── Choose model ──────────────────────────────────────────────
        if provider == "groq" and model is None:
            effective_model = _DEFAULT_MODELS["qwen27b"]
        elif provider == "qwen27b":
            provider = "groq"
            effective_model = model or _DEFAULT_MODELS["qwen27b"]
        else:
            effective_model = model or _DEFAULT_MODELS.get(provider, "gpt-4o-mini")

        effective_key = _resolve_api_key(provider, api_key)
        if not effective_key:
            logger.error("generate_visual_chat_response: No API key for provider=%s", provider)
            return "neutral", "I'm sorry — I couldn't connect to the AI service. Please check your API key in settings."

        caller = _CALLERS.get(provider)
        if caller is None:
            return "neutral", f"Unknown LLM provider: {provider}"

        system_prompt = _build_visual_chat_prompt(fused, dsmr, trajectory, allow_conflict_mention)
        messages = _build_messages(system_prompt, history, user_text)

        # ── Call LLM ─────────────────────────────────────────────────
        raw = ""
        models_to_try = [effective_model]
        if effective_model == _DEFAULT_MODELS["qwen27b"]:
            models_to_try.extend(["openai/gpt-oss-20b", "llama-3.3-70b-versatile"])

        last_exc: Optional[Exception] = None
        for attempt_model in models_to_try:
            try:
                logger.info("[VisualChat LLM] provider=%s model=%s", provider, attempt_model)
                raw = await caller(messages, effective_key, attempt_model, max_tokens=_VISUAL_MAX_TOKENS)
                if not _clean_raw(raw):
                    raise ValueError("empty response (reasoning may have used all tokens)")
                last_exc = None
                break  # success
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[VisualChat LLM] attempt with model=%s failed: %s — trying fallback.",
                    attempt_model, exc,
                )

        if last_exc is not None:
            logger.error("[VisualChat LLM] All model attempts failed: %s", last_exc)
            return "neutral", f"LLM error: {last_exc}"

        logger.debug("[VisualChat LLM] raw response: %r", raw[:200])

        emotion, sentence = _parse_emotion_sentence(raw)
        logger.info("[VisualChat LLM] parsed → emotion=%s | sentence=%r", emotion, sentence[:80])
        return emotion, sentence
