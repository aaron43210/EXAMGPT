"""
Agent 1: Router — classifies the request and resolves follow-ups.
Rules handle the clear cases for free; an LLM classifies the rest when ROUTER_LLM_ENABLED.
"""
import logging
import re
from app.agents import guardrails, prompts, tracing
from app.agents.llm_setup import chat, parse_json, LLMError
from app.agents.state import INTENTS, PipelineState
from app.agents.tracing import traced
from app.core.config import get_settings

logger = logging.getLogger(__name__)

RULES = [
    ("chitchat", re.compile(r"^\s*(hi|hello|hey|hii+|good (morning|evening|afternoon)|thanks|thank you|thx|ok(ay)?|cool|bye)\b[\s!.?]*$", re.I)),
    ("quiz", re.compile(r"\b(quiz|mcqs?|multiple[- ]choice|test me|practice (questions|test)|ask me questions)\b", re.I)),
    ("important_questions", re.compile(
        r"\b(important|most (asked|frequent|repeated)|frequently asked|expected|likely|probable|predicted?|repeated)\b.*\b(questions?|topics?)\b"
        r"|\b(pyqs?|previous year|past (year|paper)s?)\b", re.I)),
    ("summary", re.compile(r"\b(summari[sz]e|summary|overview|revision notes|revise|cheat ?sheet|key points|recap|tl;?dr)\b", re.I)),
]

_FOLLOW_UP = re.compile(
    r"\b(this|that|these|those|it|its|they|them|more|above|previous|again|elaborate|explain further)\b", re.I)


def classify_by_rules(query: str) -> str | None:
    for intent, pattern in RULES:
        if pattern.search(query):
            return intent
    return None


def resolve_search_query(query: str, history: list[dict]) -> str:
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    # Very short ("and LRU?") or pronoun-based ("explain this more") requests continue the last topic
    if last_user and (len(query.split()) <= 3 or _FOLLOW_UP.search(query)):
        return f"{query} {last_user}"
    return query


@traced("router")
def router_node(state: PipelineState) -> dict:
    query = state["user_query"].strip()
    history = state.get("chat_history", [])
    guardrails.check_input(query)

    intent, source = classify_by_rules(query), "rules"
    if intent is None and get_settings().ROUTER_LLM_ENABLED:
        recent = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in history[-4:]) or "(none)"
        try:
            data = parse_json(chat(
                [{"role": "system", "content": prompts.ROUTER.system},
                 {"role": "user", "content": prompts.ROUTER.user.format(history=recent, input=query)}],
                purpose="router", json_mode=True, max_tokens=30, temperature=0,
                prompt_version=prompts.ROUTER.id,
            ))
            if data and data.get("intent") in INTENTS:
                intent, source = data["intent"], "llm"
        except LLMError as e:
            logger.info("Router LLM unavailable, defaulting: %s", e)
    if intent is None:
        intent, source = "question", "default"

    tracing.set_intent(intent)
    logger.info("Router: intent=%s via %s", intent, source)
    return {"intent": intent, "route_source": source, "search_query": resolve_search_query(query, history)}
