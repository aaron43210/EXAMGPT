"""
Node 6: Evaluation — critiques the draft answer (optional, EVAL_ENABLED).
Conditional edge: on fail → loops back to answer_node (bounded by MAX_EVAL_RETRIES).
"""
import logging
from app.agents.state import PipelineState
from app.agents.llm_setup import chat, parse_json, LLMError
from app.core.config import get_settings

logger = logging.getLogger(__name__)

EVAL_PROMPT = """You are a strict academic reviewer. Judge the draft answer to a student's question for correctness, completeness and clarity.

Question: {query}

Draft answer:
{draft_answer}

Respond with JSON only: {{"passed": true or false, "score": integer 1-10, "feedback": "one or two sentences"}}"""


def evaluation_node(state: PipelineState) -> dict:
    settings = get_settings()
    retry_count = state.get("retry_count", 0)
    if not settings.EVAL_ENABLED:
        return {"evaluation": {}, "retry_count": retry_count}

    prompt = EVAL_PROMPT.format(query=state["user_query"], draft_answer=state.get("draft_answer", "")[:4000])
    try:
        data = parse_json(chat([{"role": "user", "content": prompt}], json_mode=True, max_tokens=200, temperature=0))
    except LLMError as e:
        # The answer already exists; never fail the request because the reviewer is unavailable
        logger.warning("Evaluation skipped: %s", e)
        data = None

    if not data:
        return {"evaluation": {}, "retry_count": retry_count}

    try:
        score = max(1, min(10, int(data.get("score", 5))))
    except (TypeError, ValueError):
        score = 5
    passed = bool(data.get("passed", True))
    evaluation = {"passed": passed, "score": score, "feedback": str(data.get("feedback", ""))[:500]}
    logger.info("Evaluation: passed=%s score=%s retry=%s", passed, score, retry_count)
    return {"evaluation": evaluation, "retry_count": retry_count + (0 if passed else 1)}


def should_retry(state: PipelineState) -> str:
    settings = get_settings()
    evaluation = state.get("evaluation") or {}
    if evaluation and not evaluation.get("passed", True) and state.get("retry_count", 0) <= settings.MAX_EVAL_RETRIES:
        return "retry"
    return "end"
