"""
Structured practice tools behind the Practice tab:
  generate_quiz — quiz master agent in JSON mode, validated MCQs
  grade_answer  — grader agent: rubric-style marking against the course material
"""
from sqlalchemy.orm import Session
from app.agents import prompts
from app.agents.llm_setup import chat, parse_json
from app.agents.nodes.retriever import gather, format_context


class InvalidModelOutput(ValueError):
    pass


def generate_quiz(db: Session, course_id: int, topic: str | None, n: int, difficulty: str) -> dict:
    query = topic or "key concepts definitions important topics"
    ctx = gather(db, course_id, query, "quiz")
    user = prompts.QUIZ_JSON.user.format(
        context=format_context(ctx), n=n, difficulty=difficulty,
        topic_clause=f" about {topic}" if topic else " covering the most important topics",
    )
    data = parse_json(chat([{"role": "system", "content": prompts.QUIZ_JSON.system},
                            {"role": "user", "content": user}],
                           purpose="quiz_master", json_mode=True, max_tokens=2500, temperature=0.4,
                           prompt_version=prompts.QUIZ_JSON.id))
    questions = []
    sources = {e["source"] for e in ctx["evidence"]}
    for q in (data or {}).get("questions") or []:
        if not isinstance(q, dict):
            continue
        options = q.get("options")
        try:
            answer_index = int(q.get("answer_index"))
        except (TypeError, ValueError):
            continue
        if (not str(q.get("question", "")).strip() or not isinstance(options, list) or len(options) != 4
                or not 0 <= answer_index < 4):
            continue
        source = str(q.get("source") or "")
        questions.append({
            "question": str(q["question"])[:1000],
            "options": [str(o)[:300] for o in options],
            "answer_index": answer_index,
            "explanation": str(q.get("explanation") or "")[:1000],
            "source": source if source in sources else None,
        })
    if not questions:
        raise InvalidModelOutput("no valid questions")
    return {"questions": questions[:n], "grounded": bool(ctx["evidence"])}


def grade_answer(db: Session, course_id: int, question: str, answer: str, max_score: int) -> dict:
    ctx = gather(db, course_id, question, "question")
    user = prompts.GRADER.user.format(context=format_context(ctx), question=question, answer=answer,
                                      max_score=max_score)
    data = parse_json(chat([{"role": "system", "content": prompts.GRADER.system},
                            {"role": "user", "content": user}],
                           purpose="grader", json_mode=True, max_tokens=1000, temperature=0,
                           prompt_version=prompts.GRADER.id))
    if not data:
        raise InvalidModelOutput("grader returned no JSON")
    try:
        score = float(data.get("score"))
    except (TypeError, ValueError):
        raise InvalidModelOutput("grader returned no score")
    as_list = lambda v: [str(x)[:300] for x in v][:8] if isinstance(v, list) else []
    return {
        "score": round(max(0.0, min(float(max_score), score)), 1),
        "max_score": max_score,
        "strengths": as_list(data.get("strengths")),
        "missing_points": as_list(data.get("missing_points")),
        "feedback": str(data.get("feedback") or "")[:2000],
        "model_answer": str(data.get("model_answer") or "")[:3000],
        "sources": sorted({e["source"] for e in ctx["evidence"]}),
    }
