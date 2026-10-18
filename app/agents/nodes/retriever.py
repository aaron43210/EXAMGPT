"""
Agent 2: Retriever — chooses which tools to call for the routed intent and assembles
evidence. Tools: hybrid search (per document type), knowledge-graph lookup, PYQ statistics.
"""
import logging
from app.agents import guardrails
from app.agents.state import PipelineState
from app.agents.tracing import span, traced
from app.core.config import get_settings
from app.db.database import SessionLocal
from app.knowledge import graph
from app.retrieval.hybrid_retriever import search, confidence

logger = logging.getLogger(__name__)

# intent -> [(doc_type, top_k, max_chars)]
SEARCH_PLAN = {
    "question": [("notes", 4, 900), ("syllabus", 2, 600), ("pyq", 3, 700)],
    "summary": [("notes", 6, 900), ("syllabus", 2, 600)],
    "quiz": [("notes", 6, 900), ("pyq", 3, 700)],
    "important_questions": [("pyq", 8, 600), ("syllabus", 3, 600)],
    "chitchat": [],
}


def gather(db, course_id: int, query: str, intent: str) -> dict:
    """Run the retrieval tools for an intent. Also used by the quiz/grading endpoints and evals."""
    settings = get_settings()
    evidence, hits_all = [], []
    for doc_type, top_k, max_chars in SEARCH_PLAN.get(intent, SEARCH_PLAN["question"]):
        with span(f"search:{doc_type}"):
            hits = search(db, course_id, query, top_k=top_k, doc_type=doc_type)
        hits_all.extend(hits)
        evidence += [{"id": h.chunk.id, "source": h.chunk.source, "page": h.chunk.page,
                      "doc_type": h.chunk.doc_type, "content": h.chunk.content[:max_chars],
                      "score": round(h.score, 4)} for h in hits]

    facts, stats = [], []
    if settings.GRAPH_RAG_ENABLED and intent != "chitchat":
        with span("graph_lookup"):
            facts = graph.neighborhood(db, course_id, graph.link_topics(db, course_id, query))
            if intent == "important_questions":
                stats = graph.pyq_stats(db, course_id)

    conf = confidence(query, hits_all)
    if not hits_all and facts:
        conf = 0.3
    return {"evidence": evidence, "graph_facts": facts, "pyq_stats": stats, "confidence": conf}


@traced("retriever")
def retriever_node(state: PipelineState) -> dict:
    db = SessionLocal()
    try:
        result = gather(db, state["course_id"], state.get("search_query") or state["user_query"],
                        state.get("intent", "question"))
    finally:
        db.close()
    guardrails.check_context(result["evidence"])
    logger.info("Retriever: %d chunks, %d graph facts, confidence=%.2f",
                len(result["evidence"]), len(result["graph_facts"]), result["confidence"])
    return result


def cite_label(e: dict) -> str:
    return f"{e['source']} p.{e['page']}" if e.get("page") else e["source"]


def format_context(state: dict) -> str:
    sections = []
    titles = {"syllabus": "Syllabus", "notes": "Notes", "pyq": "Previous year questions"}
    for doc_type, title in titles.items():
        items = [e for e in state.get("evidence", []) if e["doc_type"] == doc_type]
        if items:
            body = "\n\n".join(f"[{cite_label(e)}]\n{e['content']}" for e in items)
            sections.append(f"### {title}\n{body}")
    if state.get("graph_facts"):
        sections.append("### Topic map (knowledge graph)\n" + graph.format_facts(state["graph_facts"]))
    if state.get("pyq_stats"):
        lines = []
        for s in state["pyq_stats"]:
            years = f" — {', '.join(map(str, s['years']))}" if s["years"] else ""
            examples = "".join(f"\n    • {q['text'][:160]}" for q in s["questions"][:2])
            lines.append(f"- {s['topic']}: asked {s['count']}x{years}{examples}")
        sections.append("### Most-asked topics in past papers\n" + "\n".join(lines))
    return "\n\n".join(sections) or "(No course documents matched this request.)"
