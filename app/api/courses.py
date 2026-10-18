"""
Course API routes — CRUD operations for course management.
All endpoints are course-scoped and owner-authenticated.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import (
    Course, User, Document, Chunk, ChatHistory, PipelineRun, StudyPlan, PYQQuestion, EvaluationResult,
)
from app.schemas.course import CourseCreate, CourseResponse, CourseListResponse
from app.core.security import get_current_user
from app.api.deps import get_owned_course

router = APIRouter()

MAX_COURSES_PER_USER = 50


def _to_response(course: Course, doc_count: int) -> CourseResponse:
    return CourseResponse(
        id=course.id,
        name=course.name,
        description=course.description,
        user_id=course.user_id,
        created_at=course.created_at,
        document_count=doc_count,
    )


@router.post("/", response_model=CourseResponse, status_code=status.HTTP_201_CREATED)
def create_course(
    course_in: CourseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if db.query(Course).filter(Course.user_id == current_user.id).count() >= MAX_COURSES_PER_USER:
        raise HTTPException(status_code=400, detail=f"You can have at most {MAX_COURSES_PER_USER} courses")
    course = Course(name=course_in.name, description=course_in.description, user_id=current_user.id)
    db.add(course)
    db.commit()
    db.refresh(course)
    return _to_response(course, 0)


@router.get("/", response_model=CourseListResponse)
def list_courses(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    counts = dict(
        db.query(Document.course_id, func.count(Document.id))
        .join(Course, Course.id == Document.course_id)
        .filter(Course.user_id == current_user.id)
        .group_by(Document.course_id)
        .all()
    )
    courses = db.query(Course).filter(Course.user_id == current_user.id).order_by(Course.created_at.desc()).all()
    items = [_to_response(c, counts.get(c.id, 0)) for c in courses]
    return CourseListResponse(courses=items, total=len(items))


@router.get("/{course_id}", response_model=CourseResponse)
def get_course(course: Course = Depends(get_owned_course), db: Session = Depends(get_db)):
    doc_count = db.query(Document).filter(Document.course_id == course.id).count()
    return _to_response(course, doc_count)


@router.delete("/{course_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_course(course: Course = Depends(get_owned_course), db: Session = Depends(get_db)):
    # Delete children explicitly, in FK order, so this works on the original schema too
    doc_ids = [d for (d,) in db.query(Document.id).filter(Document.course_id == course.id)]
    pyq_ids = [p for (p,) in db.query(PYQQuestion.id).filter(PYQQuestion.course_id == course.id)]
    if pyq_ids:
        db.query(EvaluationResult).filter(EvaluationResult.question_id.in_(pyq_ids)).delete(synchronize_session=False)
    db.query(PipelineRun).filter(PipelineRun.course_id == course.id).delete(synchronize_session=False)
    db.query(ChatHistory).filter(ChatHistory.course_id == course.id).delete(synchronize_session=False)
    db.query(StudyPlan).filter(StudyPlan.course_id == course.id).delete(synchronize_session=False)
    db.query(PYQQuestion).filter(PYQQuestion.course_id == course.id).delete(synchronize_session=False)
    if doc_ids:
        db.query(PYQQuestion).filter(PYQQuestion.document_id.in_(doc_ids)).delete(synchronize_session=False)
    db.query(Chunk).filter(Chunk.course_id == course.id).delete(synchronize_session=False)
    db.query(Document).filter(Document.course_id == course.id).delete(synchronize_session=False)
    db.delete(course)
    db.commit()
