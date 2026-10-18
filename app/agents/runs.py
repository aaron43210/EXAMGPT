"""
Persisting traces (observability) and the online eval sampler.
"""
import logging
import random
from contextlib import contextmanager
from sqlalchemy.orm import Session
from app.agents.judge import judge_answer
from app.agents.llm_setup import LLMError, LLMNotConfigured, LLMRateLimited
from app.agents.tracing import Trace
from app.core.config import get_settings
from app.db.database import SessionLocal
from app.db.models import PipelineRun

logger = logging.getLogger(__name__)


def record_run(db: Session, trace: Trace, *, course_id: int, user_id: int, intent: str,
               chat_id: int | None = None, status: str = "completed", retry_count: int = 0,
               confidence: float | None = None, prompt_version: str | None = None) -> PipelineRun:
    run = PipelineRun(
        chat_id=chat_id,
        course_id=course_id,
        user_id=user_id,
        intent=intent,
        status=status,
        retry_count=retry_count,
        total_latency_ms=round(trace.elapsed_ms, 1),
        node_timings_json=trace.spans,
        model=",".join(trace.models) or None,
        prompt_version=prompt_version or (",".join(sorted(trace.prompt_versions)) or None),
        prompt_tokens=trace.prompt_tokens,
        completion_tokens=trace.completion_tokens,
        retrieval_confidence=confidence,
        guardrail_flags=trace.flags or None,
        cache_hit=trace.cache_hit,
    )
    db.add(run)
    return run


def failure_status(error: LLMError) -> str:
    if isinstance(error, LLMRateLimited):
        return "failed:rate_limited"
    if isinstance(error, LLMNotConfigured):
        return "failed:not_configured"
    return "failed:llm_unavailable"


@contextmanager
def record_llm_failures(db: Session, trace: Trace, *, course_id: int, user_id: int, intent: str | None = None):
    """Persist a failed run (rate limit / outage) before the error becomes a 429/503 response."""
    try:
        yield
    except LLMError as e:
        db.rollback()
        record_run(db, trace, course_id=course_id, user_id=user_id,
                   intent=intent or trace.intent or "unknown", status=failure_status(e))
        db.commit()
        raise


def should_sample() -> bool:
    return random.random() < get_settings().ONLINE_EVAL_SAMPLE_RATE


def score_run_in_background(run_id: int, question: str, context: str, answer: str) -> None:
    """Online eval: LLM-judge groundedness for a sampled production answer."""
    try:
        verdict = judge_answer(question, context, answer)
    except LLMError as e:
        logger.info("Online eval skipped for run %s: %s", run_id, e)
        return
    if not verdict:
        return
    db = SessionLocal()
    try:
        run = db.get(PipelineRun, run_id)
        if run is not None:
            run.groundedness = verdict["groundedness"]
            db.commit()
    finally:
        db.close()
