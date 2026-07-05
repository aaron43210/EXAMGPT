"""
Node 2: Knowledge Graph — queries Neo4j for related concepts (course-scoped).
"""
import logging
from app.agents.state import PipelineState
from app.core.neo4j_client import run_cypher

logger = logging.getLogger(__name__)


def knowledge_graph_node(state: PipelineState) -> dict:
    logger.info("Knowledge Graph node: querying Neo4j")
    course_id = state.get("course_id")
    plan = state.get("plan", [])

    # Extract topic keywords from plan
    topics = [p.replace("topic:", "") for p in plan if p.startswith("topic:")]
    if not topics:
        # Use full plan items as search terms
        topics = plan[:3]

    kg_context = []
    try:
        for topic in topics:
            records = run_cypher(
                """
                MATCH (t:Topic)-[r]-(related)
                WHERE t.course_id = $course_id
                  AND toLower(t.name) CONTAINS toLower($topic)
                RETURN t.name AS topic, type(r) AS relationship,
                       labels(related)[0] AS related_type,
                       related.name AS related_name
                LIMIT 10
                """,
                {"course_id": course_id, "topic": topic}
            )
            kg_context.extend(records)
    except Exception as e:
        logger.warning(f"Neo4j query failed (may not be available): {e}")
        # Gracefully degrade if Neo4j is not connected
        kg_context = []

    logger.info(f"KG context: {len(kg_context)} relationships found")
    return {"kg_context": kg_context}
