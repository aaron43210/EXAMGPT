"""
Semantic answer cache.

A course's grounded answers are cached with the question's embedding. A later question
with the same normalized text, or an embedding cosine >= SEMANTIC_CACHE_THRESHOLD, is answered
from cache with no LLM call. This saves Groq quota on the questions students repeat most.

Only cacheable when the answer is safe to reuse: question/summary intents, not a follow-up
(which depends on chat history), retrieval confidence high enough, no guardrail flags.
The cache for a course is cleared whenever its documents or knowledge graph change.
"""
import logging
import re
from datetime import datetime, timedelta, timezone
import numpy as np
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.db.models import AnswerCache
from app.ingestion import embedder

logger = logging.getLogger(__name__)

CACHEABLE_INTENTS = ("question", "summary")
MAX_ENTRIES_PER_COURSE = 500
_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize_query(query: str) -> str:
    return _NON_WORD.sub(" ", query.lower()).strip()


def _fresh(entry: AnswerCache, ttl: timedelta) -> bool:
    created = entry.created_at
    if created is None:
        return True
    if created.tzinfo is None:  # SQLite returns naive UTC timestamps
        created = created.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - created <= ttl


def lookup(db: Session, course_id: int, intent: str, query: str) -> AnswerCache | None:
    settings = get_settings()
    if not settings.SEMANTIC_CACHE_ENABLED or intent not in CACHEABLE_INTENTS:
        return None
    ttl = timedelta(hours=settings.SEMANTIC_CACHE_TTL_HOURS)
    entries = [e for e in db.query(AnswerCache)
               .filter(AnswerCache.course_id == course_id, AnswerCache.intent == intent)
               .order_by(AnswerCache.id.desc()).limit(MAX_ENTRIES_PER_COURSE) if _fresh(e, ttl)]
    if not entries:
        return None

    key = normalize_query(query)
    for entry in entries:
        if entry.query_key == key:
            return entry

    with_vec = [e for e in entries if e.embedding]
    if not with_vec:
        return None
    qvec = embedder.embed_query(query)
    if qvec is None:
        return None
    sims = np.array([e.embedding for e in with_vec], dtype=np.float32) @ qvec
    best = int(np.argmax(sims))
    if float(sims[best]) >= settings.SEMANTIC_CACHE_THRESHOLD:
        return with_vec[best]
    return None


def store(db: Session, *, course_id: int, intent: str, query: str, answer: str, citations: list,
          confidence: float, prompt_version: str | None) -> None:
    vectors = embedder.embed_texts([query])
    db.add(AnswerCache(
        course_id=course_id, intent=intent, query=query[:2000], query_key=normalize_query(query)[:500],
        embedding=vectors[0] if vectors else None, answer=answer, citations=citations,
        confidence=confidence, prompt_version=prompt_version,
    ))
    db.flush()
    stale = [i for (i,) in db.query(AnswerCache.id).filter(AnswerCache.course_id == course_id)
             .order_by(AnswerCache.id.desc()).offset(MAX_ENTRIES_PER_COURSE)]
    if stale:
        db.query(AnswerCache).filter(AnswerCache.id.in_(stale)).delete(synchronize_session=False)


def invalidate(db: Session, course_id: int) -> None:
    db.query(AnswerCache).filter(AnswerCache.course_id == course_id).delete(synchronize_session=False)
