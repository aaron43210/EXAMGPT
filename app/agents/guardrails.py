"""
Input/output guardrails. Cheap, deterministic checks that run on every request.
Flags are recorded on the trace (pipeline_runs.guardrail_flags) for monitoring.
"""
import re
from app.agents import prompts, tracing

_INJECTION = re.compile(
    r"(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|your|system)\s+"
    r"(instructions|prompts?|rules|messages)"
    r"|reveal\s+(your|the)\s+(system\s+)?(prompt|instructions)"
    r"|(print|show|repeat)\s+(your|the)\s+system\s+prompt"
    r"|you\s+are\s+now\s+(in\s+)?(dan|developer mode|jailbroken|an? unrestricted)"
    r"|\bjailbreak\b",
    re.IGNORECASE,
)
_CITED_FILE = re.compile(r"\[([^\[\]\n]{1,150}?\.pdf)\]", re.IGNORECASE)

MAX_ANSWER_CHARS = 12000
EMPTY_ANSWER = "I couldn't generate an answer for that. Please rephrase your question or try again."
LEAK_REPLY = "I can't share my internal instructions, but I'm happy to help with your course material."
# Distinctive fragments of system prompts; seeing them verbatim in output means a leak
_LEAK_MARKERS = [p.system[:60] for p in prompts.ALL if len(p.system) >= 60]


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text or ""))


def check_input(query: str) -> None:
    if looks_like_injection(query):
        tracing.flag("input_injection_suspected")


def check_context(evidence: list[dict]) -> None:
    if any(looks_like_injection(e.get("content", "")) for e in evidence):
        tracing.flag("context_injection_suspected")


def check_output(answer: str, evidence: list[dict]) -> str:
    answer = (answer or "").strip()
    if not answer:
        tracing.flag("empty_answer")
        return EMPTY_ANSWER
    if any(marker in answer for marker in _LEAK_MARKERS):
        tracing.flag("prompt_leak_blocked")
        return LEAK_REPLY
    sources = {e["source"].lower() for e in evidence}
    if any(name.strip().lower() not in sources for name in _CITED_FILE.findall(answer)):
        tracing.flag("unknown_citation")
    if len(answer) > MAX_ANSWER_CHARS:
        tracing.flag("answer_truncated")
        answer = answer[:MAX_ANSWER_CHARS].rsplit("\n", 1)[0] + "\n\n…"
    return answer
