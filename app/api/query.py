"""
Query API routes — submit questions and retrieve chat history.
Triggers the full 7-node LangGraph pipeline.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Course, ChatHistory, PipelineRun, User
from app.schemas.query import ChatRequest, ChatResponse, ChatHistoryItem, ChatHistoryResponse
from app.core.security import get_current_user
from app.agents.graph import run_examgpt_query

router = APIRouter()


@router.post("/{course_id}/query", response_model=ChatResponse)
def submit_query(
    course_id: int,
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Verify course ownership
    course = db.query(Course).filter(
        Course.id == course_id, Course.user_id == current_user.id
    ).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Fetch last 5 messages as in-context chat history
    recent_chats = db.query(ChatHistory).filter(
        ChatHistory.course_id == course_id,
        ChatHistory.user_id == current_user.id,
    ).order_by(ChatHistory.created_at.desc()).limit(5).all()

    chat_history = []
    for c in reversed(recent_chats):  # oldest first
        chat_history.append({"role": "user", "content": c.query})
        chat_history.append({"role": "assistant", "content": c.response})

    # Run the full pipeline
    result = run_examgpt_query(
        query=request.query,
        user_id=current_user.id,
        course_id=course_id,
        chat_history=chat_history,
    )

    # Save chat history
    chat = ChatHistory(
        user_id=current_user.id,
        course_id=course_id,
        query=request.query,
        response=result["answer"],
        citations_json=result.get("citations", []),
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)

    # Save pipeline run for observability
    pipeline_run = PipelineRun(
        chat_id=chat.id,
        course_id=course_id,
        user_id=current_user.id,
        retry_count=result.get("retry_count", 0),
        total_latency_ms=result.get("latency_ms", 0),
        status="completed" if not result.get("error") else "failed",
    )
    db.add(pipeline_run)
    db.commit()

    return ChatResponse(
        response=result["answer"],
        citations=result.get("citations", []),
        evaluation=result.get("evaluation", {}),
        retry_count=result.get("retry_count", 0),
        latency_ms=result.get("latency_ms", 0),
    )


@router.get("/{course_id}/chat-history", response_model=ChatHistoryResponse)
def get_chat_history(
    course_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    course = db.query(Course).filter(
        Course.id == course_id, Course.user_id == current_user.id
    ).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    chats = db.query(ChatHistory).filter(
        ChatHistory.course_id == course_id,
        ChatHistory.user_id == current_user.id,
    ).order_by(ChatHistory.created_at.desc()).limit(50).all()

    messages = [
        ChatHistoryItem(
            id=c.id,
            query=c.query,
            response=c.response,
            citations_json=c.citations_json,
            created_at=c.created_at,
        )
        for c in chats
    ]
    return ChatHistoryResponse(messages=messages, total=len(messages))
