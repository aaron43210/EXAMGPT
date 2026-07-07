"""
Node 5: PYQ — retrieves related Previous Year Questions + frequency signal.
"""
import logging
from app.agents.state import PipelineState
from app.retrieval.hybrid_retriever import hybrid_retriever

logger = logging.getLogger(__name__)


def pyq_node(state: PipelineState) -> dict:
    logger.info("PYQ node: retrieving previous year questions")
    query = state.get("user_query", "")
    plan = state.get("plan", [])

    search_terms = [query] + [p.replace("topic:", "") for p in plan[:3]]
    enriched_query = " ".join(search_terms)

    docs = hybrid_retriever.retrieve(
        enriched_query,
        top_k=5,
        doc_type_filter="pyq",
        course_id_filter=state.get("course_id")
    )

    related_pyqs = []
    for i, doc in enumerate(docs):
        related_pyqs.append({
            "chunk_index": i,
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "year": doc.metadata.get("year", "unknown"),
        })

    logger.info(f"PYQs retrieved: {len(related_pyqs)} questions")
    return {"related_pyqs": related_pyqs}
