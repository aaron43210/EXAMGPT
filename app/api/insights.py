"""
Practice, knowledge-graph and metrics routes:
  POST /courses/{id}/quiz         structured MCQ quiz (quiz master agent)
  POST /courses/{id}/grade        grade a written answer (grader agent)
  GET  /courses/{id}/topics       knowledge-graph topic map + most-asked topics
  GET  /courses/{id}/metrics      AI quality / cost / latency metrics for this course
  GET  /admin/metrics             same, across all users (admin role)
"""
from collections import Counter
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.agents import tracing
from app.agents.practice import generate_quiz, grade_answer, InvalidModelOutput
from app.agents.runs import record_llm_failures, record_run
from app.api.deps import get_owned_course
from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.core.security import get_current_user, require_role
from app.core.stats import mean, percentile
from app.db.database import get_db
from app.db.models import ChatHistory, Course, EvaluationResult, Feedback, PipelineRun, User
from app.knowledge.graph import topic_map
from app.schemas.practice import GradeRequest, QuizRequest

router = APIRouter()
admin_router = APIRouter()


@router.post("/{course_id}/quiz")
def create_quiz(body: QuizRequest, course: Course = Depends(get_owned_course), db: Session = Depends(get_db),
                current_user: User = Depends(get_current_user)):
    limiter.check(f"query:{current_user.id}", get_settings().RATE_LIMIT_QUERY_PER_MIN)
    trace = tracing.start_trace()
    try:
        with record_llm_failures(db, trace, course_id=course.id, user_id=current_user.id, intent="quiz_json"):
            quiz = generate_quiz(db, course.id, body.topic, body.num_questions, body.difficulty)
    except InvalidModelOutput:
        record_run(db, trace, course_id=course.id, user_id=current_user.id, intent="quiz_json", status="failed")
        db.commit()
        raise HTTPException(status_code=502, detail="The AI returned an invalid quiz. Please try again.")
    record_run(db, trace, course_id=course.id, user_id=current_user.id, intent="quiz_json")
    db.commit()
    return quiz


@router.post("/{course_id}/grade")
def grade(body: GradeRequest, course: Course = Depends(get_owned_course), db: Session = Depends(get_db),
          current_user: User = Depends(get_current_user)):
    limiter.check(f"query:{current_user.id}", get_settings().RATE_LIMIT_QUERY_PER_MIN)
    trace = tracing.start_trace()
    try:
        with record_llm_failures(db, trace, course_id=course.id, user_id=current_user.id, intent="grade"):
            result = grade_answer(db, course.id, body.question, body.answer, body.max_score)
    except InvalidModelOutput:
        record_run(db, trace, course_id=course.id, user_id=current_user.id, intent="grade", status="failed")
        db.commit()
        raise HTTPException(status_code=502, detail="The AI could not grade this answer. Please try again.")
    db.add(EvaluationResult(user_id=current_user.id, course_id=course.id, question_text=body.question,
                            user_answer=body.answer, feedback=result["feedback"],
                            score=round(result["score"]), max_score=body.max_score))
    record_run(db, trace, course_id=course.id, user_id=current_user.id, intent="grade")
    db.commit()
    return result


@router.get("/{course_id}/topics")
def get_topics(course: Course = Depends(get_owned_course), db: Session = Depends(get_db)):
    return topic_map(db, course.id)


def compute_metrics(db: Session, course_id: int | None = None, user_id: int | None = None) -> dict:
    runs_q = db.query(PipelineRun)
    fb_q = db.query(Feedback)
    ev_q = db.query(EvaluationResult)
    if course_id is not None:
        runs_q, fb_q, ev_q = (runs_q.filter(PipelineRun.course_id == course_id),
                              fb_q.filter(Feedback.course_id == course_id),
                              ev_q.filter(EvaluationResult.course_id == course_id))
    if user_id is not None:
        runs_q, fb_q, ev_q = (runs_q.filter(PipelineRun.user_id == user_id),
                              fb_q.filter(Feedback.user_id == user_id),
                              ev_q.filter(EvaluationResult.user_id == user_id))
    runs = runs_q.order_by(PipelineRun.id.desc()).limit(5000).all()
    feedback = fb_q.all()
    grades = ev_q.all()

    latencies = [r.total_latency_ms for r in runs if r.total_latency_ms is not None]
    prompt_tokens = [r.prompt_tokens or 0 for r in runs]
    completion_tokens = [r.completion_tokens or 0 for r in runs]
    grounded = [r.groundedness for r in runs if r.groundedness is not None]
    flags = Counter(f for r in runs for f in (r.guardrail_flags or []))
    step_ms: dict[str, list] = {}
    for r in runs:
        for s in r.node_timings_json or []:
            if s.get("kind") == "step":
                step_ms.setdefault(s["name"], []).append(s.get("ms", 0))
    up = sum(1 for f in feedback if f.rating > 0)
    graded = [g.score / g.max_score for g in grades if g.max_score]
    chat_runs = [r for r in runs if r.chat_id is not None]
    cache_hits = sum(1 for r in chat_runs if r.cache_hit)

    # Review queue: answers users disliked or the online judge found poorly grounded.
    # This is the data flywheel: these become new eval dataset entries.
    rating_by_chat = {f.chat_id: f.rating for f in feedback}
    grounded_by_chat = {r.chat_id: r.groundedness for r in chat_runs if r.groundedness is not None}
    flagged = {cid for cid, rating in rating_by_chat.items() if rating < 0}
    flagged |= {cid for cid, g in grounded_by_chat.items() if g < 0.5}
    review = db.query(ChatHistory).filter(ChatHistory.id.in_(flagged)).order_by(ChatHistory.id.desc()).limit(10).all() \
        if flagged else []

    return {
        "requests": {
            "total": len(runs),
            "failed": sum(1 for r in runs if (r.status or "").startswith("failed")),
            "failures_by_type": dict(Counter((r.status.split(":", 1) + ["invalid_output"])[1]
                                             for r in runs if (r.status or "").startswith("failed"))),
            "by_intent": dict(Counter(r.intent or "unknown" for r in runs)),
            "critic_retries": sum(r.retry_count or 0 for r in runs),
        },
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95), "mean": mean(latencies)},
        "agent_latency_p50_ms": {name: percentile(v, 50) for name, v in sorted(step_ms.items())},
        "tokens": {
            "prompt_total": sum(prompt_tokens),
            "completion_total": sum(completion_tokens),
            "per_request_mean": mean([p + c for p, c in zip(prompt_tokens, completion_tokens)]),
        },
        "retrieval": {"confidence_mean": mean(r.retrieval_confidence for r in runs),
                      "low_confidence_rate": mean(
                          [1.0 if r.retrieval_confidence < get_settings().LOW_CONFIDENCE_THRESHOLD else 0.0
                           for r in runs if r.retrieval_confidence is not None])},
        "online_eval": {"groundedness_mean": mean(grounded), "groundedness_p10": percentile(grounded, 10),
                        "sampled": len(grounded)},
        "feedback": {"up": up, "down": len(feedback) - up,
                     "satisfaction": round(up / len(feedback), 3) if feedback else None},
        "guardrails": dict(flags),
        "cache": {"hits": cache_hits,
                  "hit_rate": round(cache_hits / len(chat_runs), 3) if chat_runs else None,
                  "tokens_saved_est": round(cache_hits * (mean(
                      [(r.prompt_tokens or 0) + (r.completion_tokens or 0) for r in chat_runs if not r.cache_hit
                       and r.intent in ("question", "summary")]) or 0))},
        "grading": {"answers_graded": len(graded), "mean_score_pct": mean(graded)},
        "review_queue": [{"chat_id": c.id, "query": c.query[:300], "answer_preview": (c.response or "")[:300],
                          "rating": rating_by_chat.get(c.id), "groundedness": grounded_by_chat.get(c.id)}
                         for c in review],
    }


@router.get("/{course_id}/metrics")
def course_metrics(course: Course = Depends(get_owned_course), db: Session = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    return compute_metrics(db, course_id=course.id, user_id=current_user.id)


@admin_router.get("/metrics")
def admin_metrics(db: Session = Depends(get_db), _admin: User = Depends(require_role("admin"))):
    return compute_metrics(db)
