"""
Query API routes — submit questions, chat history, answer feedback.
"""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Course, ChatHistory, Feedback, PipelineRun, User
from app.schemas.query import ChatRequest, ChatResponse, ChatHistoryItem, ChatHistoryResponse, FeedbackRequest
from app.core.security import get_current_user
from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.api.deps import get_owned_course
from app.agents import guardrails, tracing
from app.agents.graph import run_examgpt_query
from app.agents.runs import record_llm_failures, record_run, should_sample, score_run_in_background
from app.retrieval import answer_cache

router = APIRouter()

HISTORY_TURNS = 3


@router.post("/{course_id}/query", response_model=ChatResponse)
def submit_query(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    course: Course = Depends(get_owned_course),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    limiter.check(f"query:{current_user.id}", get_settings().RATE_LIMIT_QUERY_PER_MIN)

    recent = db.query(ChatHistory).filter(
        ChatHistory.course_id == course.id,
        ChatHistory.user_id == current_user.id,
    ).order_by(ChatHistory.created_at.desc(), ChatHistory.id.desc()).limit(HISTORY_TURNS).all()

    chat_history = []
    for c in reversed(recent):  # oldest first
        chat_history.append({"role": "user", "content": c.query})
        chat_history.append({"role": "assistant", "content": c.response})

    trace = tracing.start_trace()
    # LLMError is recorded as a failed run, then becomes a friendly 429/503 in the global handler
    with record_llm_failures(db, trace, course_id=course.id, user_id=current_user.id):
        result = run_examgpt_query(
            query=request.query,
            user_id=current_user.id,
            course_id=course.id,
            chat_history=chat_history,
        )

    chat = ChatHistory(
        user_id=current_user.id,
        course_id=course.id,
        query=request.query,
        response=result["answer"],
        citations_json=result["citations"],
    )
    db.add(chat)
    db.flush()
    run = record_run(db, trace, course_id=course.id, user_id=current_user.id, intent=result["intent"],
                     chat_id=chat.id, retry_count=result["retry_count"], confidence=result["confidence"],
                     prompt_version=result["prompt_version"])
    if _cacheable(result, request.query, trace):
        answer_cache.store(db, course_id=course.id, intent=result["intent"], query=request.query,
                           answer=result["answer"], citations=result["citations"],
                           confidence=result["confidence"], prompt_version=result["prompt_version"])
    db.commit()

    if result["intent"] != "chitchat" and not result["from_cache"] and should_sample():
        background_tasks.add_task(score_run_in_background, run.id, request.query, result["context"], result["answer"])

    return ChatResponse(
        chat_id=chat.id,
        response=result["answer"],
        intent=result["intent"],
        confidence=result["confidence"],
        citations=result["citations"],
        evaluation=result["evaluation"],
        retry_count=result["retry_count"],
        latency_ms=round(result["latency_ms"], 1),
        guardrail_flags=trace.flags,
        cached=result["from_cache"],
        trace=trace.summary(),
    )


def _cacheable(result: dict, query: str, trace) -> bool:
    settings = get_settings()
    return (settings.SEMANTIC_CACHE_ENABLED
            and not result["from_cache"]
            and result["intent"] in answer_cache.CACHEABLE_INTENTS
            and result["search_query"] == query  # follow-ups depend on chat history
            and result["confidence"] >= settings.SEMANTIC_CACHE_MIN_CONFIDENCE
            and not trace.flags  # never cache anything a guardrail touched
            and result["answer"] not in (guardrails.EMPTY_ANSWER, guardrails.LEAK_REPLY))


@router.post("/{course_id}/chat/{chat_id}/feedback", status_code=201)
def submit_feedback(
    body: FeedbackRequest,
    chat_id: int = Path(..., ge=1),
    course: Course = Depends(get_owned_course),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    chat = db.query(ChatHistory).filter(ChatHistory.id == chat_id, ChatHistory.course_id == course.id,
                                        ChatHistory.user_id == current_user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Message not found")
    existing = db.query(Feedback).filter(Feedback.chat_id == chat.id, Feedback.user_id == current_user.id).first()
    if existing:
        existing.rating, existing.comment = body.rating, body.comment
    else:
        db.add(Feedback(chat_id=chat.id, user_id=current_user.id, course_id=course.id,
                        rating=body.rating, comment=body.comment))
    db.commit()
    return {"message": "Thanks for the feedback"}


@router.get("/{course_id}/chat-history", response_model=ChatHistoryResponse)
def get_chat_history(
    course: Course = Depends(get_owned_course),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    chats = db.query(ChatHistory).filter(
        ChatHistory.course_id == course.id,
        ChatHistory.user_id == current_user.id,
    ).order_by(ChatHistory.created_at.desc(), ChatHistory.id.desc()).limit(50).all()

    messages = [ChatHistoryItem.model_validate(c) for c in chats]
    return ChatHistoryResponse(messages=messages, total=len(messages))


@router.delete("/{course_id}/chat-history", status_code=204)
def clear_chat_history(
    course: Course = Depends(get_owned_course),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    chat_ids = [c for (c,) in db.query(ChatHistory.id).filter(
        ChatHistory.course_id == course.id, ChatHistory.user_id == current_user.id)]
    if chat_ids:
        db.query(Feedback).filter(Feedback.chat_id.in_(chat_ids)).delete(synchronize_session=False)
        db.query(PipelineRun).filter(PipelineRun.chat_id.in_(chat_ids)).delete(synchronize_session=False)
        db.query(ChatHistory).filter(ChatHistory.id.in_(chat_ids)).delete(synchronize_session=False)
    db.commit()
