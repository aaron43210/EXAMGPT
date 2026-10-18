"""
Nodes 2-4: Syllabus, Notes and PYQ retrieval — all scoped to the user's course.
"""
import logging
from app.agents.state import PipelineState
from app.db.database import SessionLocal
from app.retrieval.hybrid_retriever import retrieve

logger = logging.getLogger(__name__)


def _retrieve(state: PipelineState, doc_type: str, top_k: int, max_chars: int) -> list[dict]:
    db = SessionLocal()
    try:
        chunks = retrieve(db, state["course_id"], state.get("search_query") or state["user_query"],
                          top_k=top_k, doc_type=doc_type)
        return [{"source": c.source, "content": c.content[:max_chars], "doc_type": c.doc_type} for c in chunks]
    finally:
        db.close()


def syllabus_node(state: PipelineState) -> dict:
    found = _retrieve(state, "syllabus", top_k=2, max_chars=600)
    logger.info("Syllabus node: %d chunks", len(found))
    return {"syllabus_context": found}


def notes_node(state: PipelineState) -> dict:
    found = _retrieve(state, "notes", top_k=4, max_chars=900)
    logger.info("Notes node: %d chunks", len(found))
    return {"retrieved_notes": found}


def pyq_node(state: PipelineState) -> dict:
    found = _retrieve(state, "pyq", top_k=3, max_chars=700)
    logger.info("PYQ node: %d chunks", len(found))
    return {"related_pyqs": found}
