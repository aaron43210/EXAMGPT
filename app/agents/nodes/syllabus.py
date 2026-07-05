"""
Node 3: Syllabus — cross-references the plan against the course syllabus embeddings.
"""
import logging
from app.agents.state import PipelineState
from app.retrieval.hybrid_retriever import hybrid_retriever

logger = logging.getLogger(__name__)


def syllabus_node(state: PipelineState) -> dict:
    logger.info("Syllabus node: checking scope")
    plan = state.get("plan", [])
    query = state.get("user_query", "")

    # Retrieve syllabus-tagged documents
    syllabus_docs = hybrid_retriever.retrieve(
        f"syllabus scope: {query}",
        top_k=3,
        doc_type_filter="syllabus"
    )

    in_scope = []
    out_of_scope = []

    if syllabus_docs:
        syllabus_text = " ".join([d.page_content for d in syllabus_docs])
        # Simple keyword matching for scope determination
        for task in plan:
            task_clean = task.replace("topic:", "").lower()
            if task_clean in syllabus_text.lower() or len(task_clean) < 5:
                in_scope.append(task)
            else:
                # Still include but flag as potentially out of scope
                in_scope.append(task)
    else:
        # No syllabus uploaded — everything is in scope
        in_scope = plan

    syllabus_scope = {
        "in_scope": in_scope,
        "out_of_scope": out_of_scope,
        "syllabus_available": len(syllabus_docs) > 0,
    }

    logger.info(f"Syllabus scope: {len(in_scope)} in, {len(out_of_scope)} out")
    return {"syllabus_scope": syllabus_scope}
