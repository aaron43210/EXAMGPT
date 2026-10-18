"""
Groq LLM access. All model calls go through `chat()` so errors are mapped to a
small set of typed exceptions and the rest of the app never sees SDK details.
"""
import json
import logging
import re
from functools import lru_cache
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base class; `user_message` is safe to show to end users."""
    status_code = 503
    user_message = "The AI service is temporarily unavailable. Please try again in a minute."


class LLMRateLimited(LLMError):
    status_code = 429
    user_message = ("The AI model is receiving too many requests right now (free-tier limit). "
                    "Please wait a minute and try again.")


class LLMNotConfigured(LLMError):
    user_message = "The AI service is not configured. Please contact the administrator."


@lru_cache()
def _client():
    from groq import Groq
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        raise LLMNotConfigured("GROQ_API_KEY is not set")
    return Groq(api_key=settings.GROQ_API_KEY, timeout=settings.LLM_TIMEOUT_SECONDS, max_retries=2)


def chat(messages: list[dict], *, json_mode: bool = False, max_tokens: int = 1024,
         temperature: float = 0.2) -> str:
    """Send a chat completion to Groq and return the text content."""
    import groq
    settings = get_settings()
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = _client().chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
    except groq.RateLimitError as e:
        logger.warning("Groq rate limit: %s", e)
        raise LLMRateLimited(str(e)) from e
    except groq.AuthenticationError as e:
        logger.error("Groq authentication failed — check GROQ_API_KEY")
        raise LLMNotConfigured(str(e)) from e
    except (groq.APIConnectionError, groq.APITimeoutError, groq.APIStatusError) as e:
        logger.error("Groq API error: %s", e)
        raise LLMError(str(e)) from e
    return resp.choices[0].message.content or ""


def parse_json(text: str) -> dict | None:
    """Best-effort JSON object extraction from model output."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
