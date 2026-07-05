"""
Node 4: Notes — retrieves relevant study notes via vector similarity search (course-scoped).
"""
import logging
from app.agents.state import PipelineState
from app.retrieval.hybrid_retriever import hybrid_retriever

logger = logging.getLogger(__name__)


def notes_node(state: PipelineState) -> dict:
    logger.info("Notes node: retrieving relevant chunks")
    query = state.get("user_query", "")
    plan = state.get("plan", [])

    # Build an enriched search query from plan + original query
    search_terms = [query] + [p.replace("topic:", "") for p in plan[:3]]
    enriched_query = " ".join(search_terms)

    docs = hybrid_retriever.retrieve(
        enriched_query,
        top_k=5,
        doc_type_filter="notes"
    )

    retrieved_notes = []
    for i, doc in enumerate(docs):
        retrieved_notes.append({
            "chunk_index": i,
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "doc_type": doc.metadata.get("doc_type", "notes"),
        })

    logger.info(f"Notes retrieved: {len(retrieved_notes)} chunks")
    return {"retrieved_notes": retrieved_notes}
