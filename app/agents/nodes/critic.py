"""
Agent 7: Critic — reviews tutor drafts when retrieval confidence is low (EVAL_ENABLED).
Conditional edge: on fail, loops back to the tutor (bounded by MAX_EVAL_RETRIES).
Plus the final output guardrail and citation assembly.
"""
import logging
from app.agents import guardrails, prompts
from app.agents.llm_setup import chat, parse_json, LLMError
from app.agents.nodes.retriever import format_context
from app.agents.state import PipelineState
from app.agents.tracing import traced
from app.core.config import get_settings

logger = logging.getLogger(__name__)

CRITIQUED_INTENTS = ("question", "summary")


def needs_review(state: PipelineState) -> bool:
    settings = get_settings()
    return (settings.EVAL_ENABLED and state.get("intent") in CRITIQUED_INTENTS
            and state.get("confidence", 0) < settings.LOW_CONFIDENCE_THRESHOLD)


@traced("critic")
def critic_node(state: PipelineState) -> dict:
    retry_count = state.get("retry_count", 0)
    user = prompts.CRITIC.user.format(context=format_context(state)[:6000], input=state["user_query"],
                                      answer=state.get("draft_answer", "")[:4000])
    try:
        data = parse_json(chat(
            [{"role": "system", "content": prompts.CRITIC.system}, {"role": "user", "content": user}],
            purpose="critic", json_mode=True, max_tokens=200, temperature=0, prompt_version=prompts.CRITIC.id,
        ))
    except LLMError as e:
        # The answer already exists; never fail the request because the reviewer is unavailable
        logger.warning("Critic skipped: %s", e)
        data = None
    if not data:
        return {"evaluation": {}, "retry_count": retry_count}
    try:
        score = max(1, min(10, int(data.get("score", 5))))
    except (TypeError, ValueError):
        score = 5
    passed = bool(data.get("passed", True))
    evaluation = {"passed": passed, "score": score, "feedback": str(data.get("feedback", ""))[:500]}
    return {"evaluation": evaluation, "retry_count": retry_count + (0 if passed else 1)}


def after_critic(state: PipelineState) -> str:
    evaluation = state.get("evaluation") or {}
    if evaluation and not evaluation.get("passed", True) and state.get("retry_count", 0) <= get_settings().MAX_EVAL_RETRIES:
        return "retry"
    return "done"


@traced("output_guard")
def output_guard_node(state: PipelineState) -> dict:
    if state.get("from_cache"):
        cited = [{"source": c["source"]} for c in state.get("citations") or []]
        return {"draft_answer": guardrails.check_output(state.get("draft_answer", ""), cited)}
    evidence = state.get("evidence", [])
    answer = guardrails.check_output(state.get("draft_answer", ""), evidence)
    seen, citations = set(), []
    for e in evidence:
        key = (e["source"], e.get("page"), e["doc_type"])
        if key not in seen:
            seen.add(key)
            citations.append({"source": e["source"], "page": e.get("page"), "type": e["doc_type"]})
    return {"draft_answer": answer, "citations": citations}
