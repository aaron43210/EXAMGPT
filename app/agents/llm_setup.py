"""
LLM gateway. All model calls go through `chat()`, which provides:
  - typed errors (LLMError / LLMRateLimited / LLMNotConfigured) with user-safe messages
  - a client-side tokens-per-minute budget per model (fail fast on the free tier)
  - automatic fallback to GROQ_FALLBACK_MODEL when the primary model is rate-limited
  - token usage + latency recorded on the current trace (see tracing.py)
"""
import json
import logging
import re
import threading
import time
from collections import deque
from functools import lru_cache
from app.agents import tracing
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


class TokenBudget:
    """Sliding 60s window of tokens spent per model."""

    def __init__(self):
        self._events: dict[str, deque] = {}
        self._lock = threading.Lock()

    def _window(self, model: str) -> deque:
        events = self._events.setdefault(model, deque())
        cutoff = time.monotonic() - 60
        while events and events[0][0] < cutoff:
            events.popleft()
        return events

    def used(self, model: str) -> int:
        with self._lock:
            return sum(t for _, t in self._window(model))

    def has_room(self, model: str, tokens: int, limit: int) -> bool:
        if limit <= 0:
            return True
        with self._lock:
            used = sum(t for _, t in self._window(model))
            # Always allow a request when the window is empty, even if it alone exceeds the limit
            return used == 0 or used + tokens <= limit

    def record(self, model: str, tokens: int) -> None:
        with self._lock:
            self._window(model).append((time.monotonic(), tokens))

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


budget = TokenBudget()


def estimate_tokens(messages: list[dict], max_tokens: int) -> int:
    return sum(len(m.get("content", "")) for m in messages) // 4 + max_tokens


@lru_cache()
def _client():
    from groq import Groq
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        raise LLMNotConfigured("GROQ_API_KEY is not set")
    # Retries are handled here (fallback model) rather than by sleeping inside the SDK
    return Groq(api_key=settings.GROQ_API_KEY, timeout=settings.LLM_TIMEOUT_SECONDS, max_retries=1)


def _complete(messages: list[dict], model: str, json_mode: bool, max_tokens: int,
              temperature: float, purpose: str = "general") -> tuple[str, dict]:
    """One Groq call. Returns (text, usage). Tests replace this function."""
    import groq
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    try:
        resp = _client().chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs,
        )
    except groq.RateLimitError as e:
        raise LLMRateLimited(str(e)) from e
    except groq.AuthenticationError as e:
        logger.error("Groq authentication failed — check GROQ_API_KEY")
        raise LLMNotConfigured(str(e)) from e
    except (groq.APIConnectionError, groq.APITimeoutError, groq.APIStatusError) as e:
        logger.error("Groq API error (%s, %s): %s", model, purpose, e)
        raise LLMError(str(e)) from e
    usage = resp.usage
    return resp.choices[0].message.content or "", {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }


def chat(messages: list[dict], *, purpose: str = "general", json_mode: bool = False,
         max_tokens: int = 1024, temperature: float = 0.2, model: str | None = None,
         prompt_version: str | None = None) -> str:
    """Send a chat completion and return the text content."""
    settings = get_settings()
    if model:
        candidates = [(model, 0)]
    else:
        candidates = [(settings.GROQ_MODEL, settings.GROQ_TPM_LIMIT)]
        if settings.GROQ_FALLBACK_MODEL and settings.GROQ_FALLBACK_MODEL != settings.GROQ_MODEL:
            candidates.append((settings.GROQ_FALLBACK_MODEL, settings.GROQ_FALLBACK_TPM_LIMIT))

    estimate = estimate_tokens(messages, max_tokens)
    last_error: LLMError = LLMRateLimited("token budget exhausted")
    for name, limit in candidates:
        if not budget.has_room(name, estimate, limit):
            logger.info("Local token budget full for %s, trying next model", name)
            continue
        start = time.perf_counter()
        try:
            text, usage = _complete(messages, name, json_mode, max_tokens, temperature, purpose)
        except LLMRateLimited as e:
            logger.warning("Rate limited on %s (%s)", name, purpose)
            budget.record(name, max(limit, estimate))  # block this model for the rest of the window
            last_error = e
            continue
        spent = (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)) or estimate
        budget.record(name, spent)
        tracing.record_llm(purpose, name, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                           (time.perf_counter() - start) * 1000, prompt_version)
        return text
    raise last_error


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
