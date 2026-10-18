"""
ExamGPT multi-agent workflow (LangGraph).

  router ─┬─ chitchat ─────────────────────────────────────────────────────┐
          ├─ cache_lookup ── hit ──────────────────────────────────────────┤
          │        └── miss ─┐                                             │
          └──────────────────┴─ retriever ─┬─ tutor ──▶ [critic] ──────────┤
                                           ├─ quiz_master ─────────────────┤
                                           └─ exam_analyst ────────────────┤
                                                critic ─ retry ─▶ tutor    ▼
                                                                    output_guard ──▶ END
"""
import logging
from langgraph.graph import StateGraph, END
from app.agents import tracing
from app.agents.nodes.cache import cache_lookup_node, is_cacheable_request
from app.agents.nodes.critic import critic_node, after_critic, needs_review, output_guard_node
from app.agents.nodes.retriever import retriever_node, format_context
from app.agents.nodes.router import router_node
from app.agents.nodes.specialists import analyst_node, chitchat_node, quiz_node, tutor_node
from app.agents.state import PipelineState

logger = logging.getLogger(__name__)

SPECIALIST = {"question": "tutor", "summary": "tutor", "quiz": "quiz_master",
              "important_questions": "exam_analyst"}

AGENTS = ["router", "retriever", "tutor", "quiz_master", "exam_analyst", "chitchat", "critic"]


def _after_router(state: PipelineState) -> str:
    if state["intent"] == "chitchat":
        return "chitchat"
    return "cache_lookup" if is_cacheable_request(state) else "retriever"


workflow = StateGraph(PipelineState)
workflow.add_node("router", router_node)
workflow.add_node("cache_lookup", cache_lookup_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("tutor", tutor_node)
workflow.add_node("quiz_master", quiz_node)
workflow.add_node("exam_analyst", analyst_node)
workflow.add_node("chitchat", chitchat_node)
workflow.add_node("critic", critic_node)
workflow.add_node("output_guard", output_guard_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", _after_router,
                               {"chitchat": "chitchat", "cache_lookup": "cache_lookup", "retriever": "retriever"})
workflow.add_conditional_edges("cache_lookup", lambda s: "output_guard" if s.get("from_cache") else "retriever",
                               {"output_guard": "output_guard", "retriever": "retriever"})
workflow.add_conditional_edges("retriever", lambda s: SPECIALIST.get(s["intent"], "tutor"),
                               {v: v for v in set(SPECIALIST.values())})
workflow.add_conditional_edges("tutor", lambda s: "critic" if needs_review(s) else "output_guard",
                               {"critic": "critic", "output_guard": "output_guard"})
workflow.add_conditional_edges("critic", after_critic, {"retry": "tutor", "done": "output_guard"})
for name in ("quiz_master", "exam_analyst", "chitchat"):
    workflow.add_edge(name, "output_guard")
workflow.add_edge("output_guard", END)

examgpt_graph = workflow.compile()


def run_examgpt_query(query: str, user_id: int, course_id: int, chat_history: list | None = None) -> dict:
    """Run the agents. LLM errors (LLMError) propagate to the caller."""
    trace = tracing.current() or tracing.start_trace()
    final = examgpt_graph.invoke({
        "user_id": user_id,
        "course_id": course_id,
        "user_query": query,
        "chat_history": chat_history or [],
        "retry_count": 0,
    })
    logger.info("Pipeline completed in %.0fms (intent=%s, cached=%s)",
                trace.elapsed_ms, final.get("intent"), bool(final.get("from_cache")))
    return {
        "answer": final.get("draft_answer", ""),
        "citations": final.get("citations", []),
        "evaluation": final.get("evaluation", {}),
        "retry_count": final.get("retry_count", 0),
        "intent": final.get("intent"),
        "search_query": final.get("search_query", query),
        "from_cache": bool(final.get("from_cache")),
        "confidence": final.get("confidence", 0.0),
        "prompt_version": final.get("prompt_version"),
        "context": format_context(final),
        "evidence": final.get("evidence", []),
        "latency_ms": trace.elapsed_ms,
        "trace": trace,
    }
