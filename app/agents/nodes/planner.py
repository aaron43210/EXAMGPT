"""
Node 1: Planner — formulates an execution plan from the user's query + course context.
"""
import json
import logging
from app.agents.state import PipelineState
from app.agents.llm_setup import get_llm

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """You are a study planning agent for ExamGPT. Given a student's question about their course material, create a structured execution plan.

The plan should identify:
1. The main topics/concepts to look up
2. Whether to check the syllabus for scope
3. Whether to search notes for relevant content
4. Whether to check previous year questions for patterns

## Recent Chat History
{chat_history}

## Student Query
{query}

Respond with a JSON object:
{{
  "subtasks": ["list of specific subtasks to execute"],
  "topics": ["list of key topics to search for"],
  "check_syllabus": true/false,
  "check_notes": true/false,
  "check_pyqs": true/false
}}

Respond ONLY with the JSON, no other text."""


def planner_node(state: PipelineState) -> dict:
    logger.info("Planner node: formulating plan")
    llm = get_llm()
    query = state["user_query"]

    # Format chat history
    history = state.get("chat_history", [])
    if history:
        history_str = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in history
        )
    else:
        history_str = "No previous messages."

    response = llm.invoke(PLANNER_PROMPT.format(query=query, chat_history=history_str))
    response_text = response.content if hasattr(response, "content") else str(response)

    try:
        plan_data = json.loads(response_text.strip())
        subtasks = plan_data.get("subtasks", [query])
        topics = plan_data.get("topics", [])
    except (json.JSONDecodeError, AttributeError):
        subtasks = [query]
        topics = []

    plan = subtasks + [f"topic:{t}" for t in topics]
    logger.info(f"Planner output: {len(plan)} subtasks")

    return {"plan": plan}
