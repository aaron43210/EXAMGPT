"""
Semantic-cache lookup step (between the router and the retriever).
A hit skips retrieval and the LLM entirely; see app/retrieval/answer_cache.py.
"""
import logging
from app.agents import tracing
from app.agents.state import PipelineState
from app.agents.tracing import traced
from app.db.database import SessionLocal
from app.retrieval import answer_cache

logger = logging.getLogger(__name__)


def is_cacheable_request(state: PipelineState) -> bool:
    """Follow-ups depend on chat history, so only self-contained requests use the cache."""
    return (state.get("intent") in answer_cache.CACHEABLE_INTENTS
            and state.get("search_query", state["user_query"]) == state["user_query"])


@traced("cache_lookup")
def cache_lookup_node(state: PipelineState) -> dict:
    db = SessionLocal()
    try:
        hit = answer_cache.lookup(db, state["course_id"], state["intent"], state["user_query"])
        if hit is None:
            return {"from_cache": False}
        hit.hits = (hit.hits or 0) + 1
        db.commit()
        tracing.mark_cache_hit()
        logger.info("Semantic cache hit (entry %s)", hit.id)
        return {"from_cache": True, "draft_answer": hit.answer, "citations": hit.citations or [],
                "confidence": hit.confidence or 0.0, "prompt_version": hit.prompt_version}
    finally:
        db.close()
