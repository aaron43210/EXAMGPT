"""
Query API routes — submit questions and retrieve chat history.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Course, ChatHistory, PipelineRun, User
from app.schemas.query import ChatRequest, ChatResponse, ChatHistoryItem, ChatHistoryResponse
from app.core.security import get_current_user
from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.api.deps import get_owned_course
from app.agents.graph import run_examgpt_query

router = APIRouter()

HISTORY_TURNS = 3


@router.post("/{course_id}/query", response_model=ChatResponse)
def submit_query(
    request: ChatRequest,
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

    # LLMError propagates to the global handler, which returns a friendly 429/503
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
    db.add(PipelineRun(
        chat_id=chat.id,
        course_id=course.id,
        user_id=current_user.id,
        retry_count=result["retry_count"],
        total_latency_ms=result["latency_ms"],
        status="completed",
    ))
    db.commit()

    return ChatResponse(
        response=result["answer"],
        citations=result["citations"],
        evaluation=result["evaluation"],
        retry_count=result["retry_count"],
        latency_ms=result["latency_ms"],
    )


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
        db.query(PipelineRun).filter(PipelineRun.chat_id.in_(chat_ids)).delete(synchronize_session=False)
        db.query(ChatHistory).filter(ChatHistory.id.in_(chat_ids)).delete(synchronize_session=False)
    db.commit()
