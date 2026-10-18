"""
Pipeline state definition for the ExamGPT multi-agent LangGraph workflow.
"""
from typing import TypedDict


class PipelineState(TypedDict, total=False):
    # ── Input ───────────────────────────────────────────────────
    user_id: int
    course_id: int
    user_query: str
    chat_history: list[dict]  # [{"role": "user"/"assistant", "content": "..."}]

    # ── Planner output ──────────────────────────────────────────
    search_query: str         # query enriched with follow-up context

    # ── Retrieval outputs ───────────────────────────────────────
    syllabus_context: list[dict]
    retrieved_notes: list[dict]
    related_pyqs: list[dict]

    # ── Answer / evaluation ─────────────────────────────────────
    draft_answer: str
    citations: list[dict]
    evaluation: dict          # {"passed": bool, "score": int, "feedback": str}
    retry_count: int
