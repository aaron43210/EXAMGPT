"""
Pipeline state definition for the ExamGPT multi-agent LangGraph workflow.
Matches PRD §8.4.
"""
from typing import TypedDict


class PipelineState(TypedDict, total=False):
    # ── Input ───────────────────────────────────────────────────
    user_id: int
    course_id: int
    user_query: str

    # ── Planner output ──────────────────────────────────────────
    plan: list[str]           # subtask list from planner

    # ── Knowledge Graph output ──────────────────────────────────
    kg_context: list[dict]    # entities/relationships from Neo4j

    # ── Syllabus output ─────────────────────────────────────────
    syllabus_scope: dict      # {"in_scope": [...], "out_of_scope": [...]}

    # ── Notes retrieval output ──────────────────────────────────
    retrieved_notes: list[dict]  # top-k chunks with source metadata

    # ── PYQ retrieval output ────────────────────────────────────
    related_pyqs: list[dict]  # PYQs + frequency signal

    # ── Answer output ───────────────────────────────────────────
    draft_answer: str
    citations: list[dict]     # source references

    # ── Evaluation output ───────────────────────────────────────
    evaluation: dict          # {"passed": bool, "feedback": str}
    retry_count: int

    # ── Error tracking ──────────────────────────────────────────
    error: str
