"""
Node 1: Planner — builds the retrieval query. Rule-based (no LLM call) so each
question spends its Groq budget on the answer itself.
"""
import logging
import re
from app.agents.state import PipelineState

logger = logging.getLogger(__name__)

_FOLLOW_UP = re.compile(
    r"\b(this|that|these|those|it|its|they|them|more|above|previous|again|elaborate|explain further)\b",
    re.IGNORECASE,
)


def planner_node(state: PipelineState) -> dict:
    query = state["user_query"].strip()
    history = state.get("chat_history", [])
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")

    # Short or pronoun-heavy follow-ups ("explain this more") need the previous topic
    if last_user and (len(query.split()) <= 6 or _FOLLOW_UP.search(query)):
        search_query = f"{query} {last_user}"
    else:
        search_query = query

    logger.info("Planner: search query built (%d chars)", len(search_query))
    return {"search_query": search_query}
