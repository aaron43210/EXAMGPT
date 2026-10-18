"""
Lightweight request tracing. Every agent step and every LLM call appends a span to the
current Trace (held in a ContextVar), which is persisted to pipeline_runs for metrics.
"""
import functools
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class Trace:
    spans: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    prompt_versions: set[str] = field(default_factory=set)
    cache_hit: bool = False
    intent: str | None = None
    started: float = field(default_factory=time.perf_counter)

    @property
    def prompt_tokens(self) -> int:
        return sum(s.get("prompt_tokens", 0) for s in self.spans)

    @property
    def completion_tokens(self) -> int:
        return sum(s.get("completion_tokens", 0) for s in self.spans)

    @property
    def models(self) -> list[str]:
        return sorted({s["model"] for s in self.spans if s.get("model")})

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000

    def flag(self, name: str) -> None:
        if name not in self.flags:
            self.flags.append(name)

    def summary(self) -> dict:
        return {
            "steps": [{k: v for k, v in s.items() if k in ("name", "kind", "ms", "model")} for s in self.spans],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "models": self.models,
            "cache_hit": self.cache_hit,
        }


_current: ContextVar[Trace | None] = ContextVar("examgpt_trace", default=None)


def start_trace() -> Trace:
    trace = Trace()
    _current.set(trace)
    return trace


def current() -> Trace | None:
    return _current.get()


def record_llm(purpose: str, model: str, prompt_tokens: int, completion_tokens: int, ms: float,
               prompt_version: str | None = None) -> None:
    trace = _current.get()
    if trace is None:
        return
    trace.spans.append({
        "name": purpose, "kind": "llm", "model": model, "ms": round(ms, 1),
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
    })
    if prompt_version:
        trace.prompt_versions.add(prompt_version)


def set_intent(intent: str) -> None:
    trace = _current.get()
    if trace is not None:
        trace.intent = intent


def mark_cache_hit() -> None:
    trace = _current.get()
    if trace is not None:
        trace.cache_hit = True


def flag(name: str) -> None:
    trace = _current.get()
    if trace is not None:
        trace.flag(name)


@contextmanager
def span(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        trace = _current.get()
        if trace is not None:
            trace.spans.append({"name": name, "kind": "step", "ms": round((time.perf_counter() - start) * 1000, 1)})


def traced(name: str):
    """Decorator for LangGraph nodes."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            with span(name):
                return fn(*args, **kwargs)
        return inner
    return wrap
