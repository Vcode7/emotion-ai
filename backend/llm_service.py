"""
llm_service.py — Multi-provider LLM response generation
=========================================================

Builds an emotion-aware system prompt from the DMESR pipeline output
and routes the request to one of three providers:

    • **OpenAI**  — ``gpt-4o-mini`` via the official async client
    • **Groq**    — ``llama-3.3-70b-versatile`` via OpenAI-compatible endpoint
    • **Gemini**  — ``gemini-2.0-flash`` via ``google-genai``

The LLM is instructed to return a JSON array of response segments:

    [{"text": "…", "emotion": "compassion"}, …]

Supported emotion labels for the response:
    compassion, supportive, encouraging, concerned,
    neutral, happy, thinking, sad
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Valid emotion labels the LLM may use in its response segments.
VALID_EMOTIONS = frozenset(
    ["compassion", "supportive", "encouraging", "concerned",
     "neutral", "happy", "thinking", "sad"]
)


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(
    emotion_state: Dict[str, float],
    llm_context: Dict[str, Any],
) -> str:
    """
    Build the system prompt that injects real-time emotion analysis into
    the LLM's instruction context.
    """
    velocity = llm_context.get("emotion_velocity") or {}
    trends = llm_context.get("trend_labels") or {}
    summary = llm_context.get("conversation_summary") or {}

    # Format current emotion state
    state_lines = "\n".join(
        f"  • {dim}: {val:+.3f}" for dim, val in emotion_state.items()
    )

    # Format velocity
    if velocity:
        vel_lines = "\n".join(
            f"  • {dim}: {val:+.4f} ({trends.get(dim, '')})"
            for dim, val in velocity.items()
        )
    else:
        vel_lines = "  (first turn — no velocity yet)"

    # Format conversation summary
    if summary:
        sum_parts = []
        for dim, stats in summary.items():
            mean_val = stats.get("mean", 0.0)
            std_val = stats.get("std", 0.0)
            sum_parts.append(f"  • {dim}: mean={mean_val:+.3f}, std={std_val:.3f}")
        summary_lines = "\n".join(sum_parts)
    else:
        summary_lines = "  (no history yet)"

    return f"""You are an empathetic AI companion. You have access to real-time emotion analysis of the user via the DMESR Emotion-AI system.

## Current User Emotion State (11-dimensional, range [-1, +1]):
{state_lines}

## Emotion Velocity (change since last turn):
{vel_lines}

## Conversation Emotion Summary:
{summary_lines}

## Instructions:
- Adapt your tone, language, and emotional register to match the user's detected emotional state.
- If empathy_needed is high, lead with compassion and validation before offering advice.
- If stress is high and confidence is low, be reassuring and gently encouraging.
- If the user seems positive, match their energy.
- Keep responses concise but warm — 1 to 3 sentences per segment.

## Response Format:
Respond ONLY with a JSON array. Each element must have:
  "text" — your response text for that segment
  "emotion" — one of: compassion, supportive, encouraging, concerned, neutral, happy, thinking, sad

Example:
[{{"text": "I hear you, that sounds really tough.", "emotion": "compassion"}}, {{"text": "Let's think about what might help.", "emotion": "supportive"}}]

Respond ONLY with the JSON array. No markdown, no code fences, no explanation."""


# ─────────────────────────────────────────────────────────────────────────────
#  Response parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_llm_response(raw: str) -> List[Dict[str, str]]:
    """
    Parse the raw LLM output into a list of ``{text, emotion}`` dicts.

    Handles:
      - Clean JSON arrays
      - JSON wrapped in markdown code fences
      - Completely malformed output (falls back to neutral)
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            # Validate each segment
            result: List[Dict[str, str]] = []
            for item in parsed:
                if isinstance(item, dict) and "text" in item:
                    emotion = item.get("emotion", "neutral")
                    if emotion not in VALID_EMOTIONS:
                        emotion = "neutral"
                    result.append({"text": str(item["text"]), "emotion": emotion})
            if result:
                return result
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: treat the entire response as a single neutral segment
    logger.warning("LLM response was not valid JSON — using raw text fallback.")
    return [{"text": raw.strip() or "I'm here to help.", "emotion": "neutral"}]


# ─────────────────────────────────────────────────────────────────────────────
#  Provider-specific callers
# ─────────────────────────────────────────────────────────────────────────────

async def _call_openai(
    system_prompt: str,
    user_text: str,
    api_key: str,
    model: str,
) -> str:
    """Call the OpenAI Chat Completions API."""
    import openai

    client = openai.AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


async def _call_groq(
    system_prompt: str,
    user_text: str,
    api_key: str,
    model: str,
) -> str:
    """Call Groq via the OpenAI-compatible endpoint."""
    import openai

    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


async def _call_gemini(
    system_prompt: str,
    user_text: str,
    api_key: str,
    model: str,
) -> str:
    """Call Google Gemini via the google-genai SDK."""
    from google import genai

    client = genai.Client(api_key=api_key)

    # Gemini uses a combined prompt rather than system/user roles in the
    # genai SDK's simple generate_content path.
    combined_prompt = f"{system_prompt}\n\nUser message:\n{user_text}"

    response = client.models.generate_content(
        model=model,
        contents=combined_prompt,
    )
    return response.text or ""


# ─────────────────────────────────────────────────────────────────────────────
#  Public Service Class
# ─────────────────────────────────────────────────────────────────────────────

# Default models per provider
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
}


class LLMService:
    """
    Stateless multi-provider LLM service.

    Builds an emotion-aware system prompt from the DMESR pipeline output,
    routes the request to the selected provider, and parses the response
    into structured segments.
    """

    async def generate_response(
        self,
        user_text: str,
        emotion_state: Dict[str, float],
        llm_context: Dict[str, Any],
        provider: str,
        api_key: Optional[str],
        model: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Generate an emotion-aware LLM response.

        Args:
            user_text:      The user's input text.
            emotion_state:  11-D emotion dict from DMESR.
            llm_context:    Full context dict from ``EmotionMemory.to_llm_context()``.
            provider:       ``"openai"``, ``"groq"``, or ``"gemini"``.
            api_key:        Provider API key (optional, falls back to env/config).
            model:          Optional model override.

        Returns:
            List of ``{"text": str, "emotion": str}`` segments.
        """
        # Resolve model name
        effective_model = model or _DEFAULT_MODELS.get(provider, "gpt-4o-mini")

        # Resolve API key fallback
        effective_key = api_key
        if not effective_key:
            from config import GROQ_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
            if provider == "groq":
                effective_key = GROQ_API_KEY
            elif provider == "openai":
                effective_key = OPENAI_API_KEY
            elif provider == "gemini":
                effective_key = GEMINI_API_KEY

        if not effective_key:
            logger.error("No API key provided for provider: %s", provider)
            return [{
                "text": f"LLM error: No API key provided for '{provider}'. Please configure it in settings or the .env file.",
                "emotion": "neutral"
            }]

        # Build prompt
        system_prompt = _build_system_prompt(emotion_state, llm_context)

        # Route to provider
        try:
            if provider == "openai":
                raw = await _call_openai(system_prompt, user_text, effective_key, effective_model)
            elif provider == "groq":
                raw = await _call_groq(system_prompt, user_text, effective_key, effective_model)
            elif provider == "gemini":
                raw = await _call_gemini(system_prompt, user_text, effective_key, effective_model)
            else:
                logger.error("Unknown LLM provider: %s", provider)
                return [{"text": f"Unknown LLM provider: {provider}", "emotion": "neutral"}]

        except Exception as exc:
            logger.exception("LLM API call failed (%s / %s).", provider, effective_model)
            return [{"text": f"LLM error: {exc}", "emotion": "neutral"}]

        return _parse_llm_response(raw)
