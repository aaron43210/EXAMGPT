"""
ExamGPT LangGraph — multi-agent pipeline with an optional self-evaluation retry loop.

planner → syllabus → notes → pyq → answer → evaluation ─┬─ end
                                    ▲                   │
                                    └────── retry ──────┘
"""
import logging
import time
from langgraph.graph import StateGraph, END
from app.agents.state import PipelineState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retrieval import syllabus_node, notes_node, pyq_node
from app.agents.nodes.answer import answer_node
from app.agents.nodes.evaluation import evaluation_node, should_retry

logger = logging.getLogger(__name__)

workflow = StateGraph(PipelineState)
workflow.add_node("planner", planner_node)
workflow.add_node("syllabus", syllabus_node)
workflow.add_node("notes", notes_node)
workflow.add_node("pyq", pyq_node)
workflow.add_node("answer", answer_node)
workflow.add_node("evaluation", evaluation_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "syllabus")
workflow.add_edge("syllabus", "notes")
workflow.add_edge("notes", "pyq")
workflow.add_edge("pyq", "answer")
workflow.add_edge("answer", "evaluation")
workflow.add_conditional_edges("evaluation", should_retry, {"retry": "answer", "end": END})

examgpt_graph = workflow.compile()


def run_examgpt_query(query: str, user_id: int, course_id: int, chat_history: list | None = None) -> dict:
    """Run the pipeline. LLM errors (LLMError) propagate to the caller."""
    start = time.time()
    final_state = examgpt_graph.invoke({
        "user_id": user_id,
        "course_id": course_id,
        "user_query": query,
        "chat_history": chat_history or [],
        "retry_count": 0,
    })
    elapsed_ms = (time.time() - start) * 1000
    logger.info("Pipeline completed in %.0fms", elapsed_ms)
    return {
        "answer": final_state.get("draft_answer", ""),
        "citations": final_state.get("citations", []),
        "evaluation": final_state.get("evaluation", {}),
        "retry_count": final_state.get("retry_count", 0),
        "latency_ms": elapsed_ms,
    }
