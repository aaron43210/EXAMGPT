"""
ExamGPT LangGraph — Full 7-node multi-agent pipeline with self-evaluation retry loop.
Matches PRD §8.
"""
import logging
import time
from langgraph.graph import StateGraph, END
from app.agents.state import PipelineState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.knowledge_graph import knowledge_graph_node
from app.agents.nodes.syllabus import syllabus_node
from app.agents.nodes.notes import notes_node
from app.agents.nodes.pyq import pyq_node
from app.agents.nodes.answer import answer_node
from app.agents.nodes.evaluation import evaluation_node, should_retry

logger = logging.getLogger(__name__)

# ── Build the Graph ──────────────────────────────────────────────────────

workflow = StateGraph(PipelineState)

# Add all 7 nodes
workflow.add_node("planner", planner_node)
workflow.add_node("knowledge_graph", knowledge_graph_node)
workflow.add_node("syllabus", syllabus_node)
workflow.add_node("notes", notes_node)
workflow.add_node("pyq", pyq_node)
workflow.add_node("answer", answer_node)
workflow.add_node("evaluation", evaluation_node)

# Define edges — linear flow
workflow.set_entry_point("planner")
workflow.add_edge("planner", "knowledge_graph")
workflow.add_edge("knowledge_graph", "syllabus")
workflow.add_edge("syllabus", "notes")
workflow.add_edge("notes", "pyq")
workflow.add_edge("pyq", "answer")
workflow.add_edge("answer", "evaluation")

# Conditional edge: evaluation → retry(answer) or → END
workflow.add_conditional_edges(
    "evaluation",
    should_retry,
    {
        "retry": "answer",
        "end": END,
    }
)

# Compile
examgpt_graph = workflow.compile()


# ── Execution Function ───────────────────────────────────────────────────

def run_examgpt_query(
    query: str,
    user_id: int = 0,
    course_id: int = 0,
    chat_history: list = None,
) -> dict:
    """
    Main entry point: runs the full 7-node pipeline.
    Returns dict with draft_answer, citations, evaluation, and timing.
    """
    logger.info(f"Starting pipeline for user={user_id}, course={course_id}")
    start = time.time()

    initial_state: PipelineState = {
        "user_id": user_id,
        "course_id": course_id,
        "user_query": query,
        "chat_history": chat_history or [],
        "plan": [],
        "kg_context": [],
        "syllabus_scope": {},
        "retrieved_notes": [],
        "related_pyqs": [],
        "draft_answer": "",
        "citations": [],
        "evaluation": {},
        "retry_count": 0,
        "error": "",
    }

    try:
        final_state = examgpt_graph.invoke(initial_state)
        elapsed_ms = (time.time() - start) * 1000
        logger.info(f"Pipeline completed in {elapsed_ms:.0f}ms")

        return {
            "answer": final_state.get("draft_answer", ""),
            "citations": final_state.get("citations", []),
            "evaluation": final_state.get("evaluation", {}),
            "retry_count": final_state.get("retry_count", 0),
            "latency_ms": elapsed_ms,
        }
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        logger.error(f"Pipeline failed after {elapsed_ms:.0f}ms: {e}", exc_info=True)
        return {
            "answer": f"Error: {str(e)}",
            "citations": [],
            "evaluation": {"passed": False, "feedback": str(e)},
            "retry_count": 0,
            "latency_ms": elapsed_ms,
            "error": str(e),
        }
