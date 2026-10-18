"""
LLM-as-judge used by both the online sampler and the offline eval harness.

groundedness = supported_claims / total_claims      (a.k.a. faithfulness)
relevance    = (rating - 1) / 4                     (1-5 rating mapped to 0-1)

The judge runs on JUDGE_MODEL, a different (larger) model than the answer model, to reduce
self-preference bias.
"""
import logging
from app.agents import prompts
from app.agents.llm_setup import chat, parse_json
from app.core.config import get_settings

logger = logging.getLogger(__name__)


def judge_answer(question: str, context: str, answer: str) -> dict | None:
    settings = get_settings()
    data = parse_json(chat(
        [{"role": "system", "content": prompts.JUDGE.system},
         {"role": "user", "content": prompts.JUDGE.user.format(
             input=question[:2000], context=context[:8000] or "(no context retrieved)", answer=answer[:6000])}],
        purpose="judge", json_mode=True, max_tokens=1200, temperature=0,
        model=settings.JUDGE_MODEL or None, prompt_version=prompts.JUDGE.id,
    ))
    if not data:
        return None
    claims = [c for c in data.get("claims") or [] if isinstance(c, dict)]
    supported = sum(1 for c in claims if c.get("supported") is True)
    try:
        rating = max(1, min(5, int(data.get("relevance", 3))))
    except (TypeError, ValueError):
        rating = 3
    return {
        "groundedness": round(supported / len(claims), 3) if claims else 1.0,
        "relevance": round((rating - 1) / 4, 3),
        "claims_total": len(claims),
        "claims_supported": supported,
        "unsupported_claims": [str(c.get("claim", ""))[:300] for c in claims if c.get("supported") is not True],
    }
