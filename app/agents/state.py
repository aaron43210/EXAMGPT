"""
Shared state passed between ExamGPT agents in the LangGraph workflow.
"""
from typing import TypedDict

INTENTS = ("question", "summary", "quiz", "important_questions", "chitchat")


class PipelineState(TypedDict, total=False):
    # ── Input ───────────────────────────────────────────────────
    user_id: int
    course_id: int
    user_query: str
    chat_history: list[dict]  # [{"role": "user"/"assistant", "content": "..."}]

    # ── Router agent ────────────────────────────────────────────
    intent: str               # one of INTENTS
    route_source: str         # "rules" | "llm" | "default"
    search_query: str         # query enriched with follow-up context
    from_cache: bool          # answered from the semantic cache (no retrieval / LLM call)

    # ── Retriever agent ─────────────────────────────────────────
    evidence: list[dict]      # [{"id", "source", "page", "doc_type", "content", "score"}]
    graph_facts: list[dict]   # topic neighbourhoods from the knowledge graph
    pyq_stats: list[dict]     # most-asked topics (important_questions intent)
    confidence: float         # 0-1 retrieval confidence

    # ── Specialist agents / critic ──────────────────────────────
    draft_answer: str
    prompt_version: str
    citations: list[dict]
    evaluation: dict          # critic verdict {"passed", "score", "feedback"}
    retry_count: int
