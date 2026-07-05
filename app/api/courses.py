"""
Course API routes — CRUD operations for course management.
All endpoints are course-scoped and owner-authenticated.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Course, User, Document
from app.schemas.course import CourseCreate, CourseResponse, CourseListResponse
from app.core.security import get_current_user

router = APIRouter()


@router.post("/", response_model=CourseResponse, status_code=status.HTTP_201_CREATED)
def create_course(
    course_in: CourseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    course = Course(
        name=course_in.name,
        description=course_in.description,
        user_id=current_user.id,
    )
    db.add(course)
    db.commit()
    db.refresh(course)
    return CourseResponse(
        id=course.id,
        name=course.name,
        description=course.description,
        user_id=course.user_id,
        created_at=course.created_at,
        document_count=0,
    )


@router.get("/", response_model=CourseListResponse)
def list_courses(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    courses = db.query(Course).filter(Course.user_id == current_user.id).all()
    course_responses = []
    for c in courses:
        doc_count = db.query(Document).filter(Document.course_id == c.id).count()
        course_responses.append(CourseResponse(
            id=c.id,
            name=c.name,
            description=c.description,
            user_id=c.user_id,
            created_at=c.created_at,
            document_count=doc_count,
        ))
    return CourseListResponse(courses=course_responses, total=len(course_responses))


@router.get("/{course_id}", response_model=CourseResponse)
def get_course(
    course_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    course = db.query(Course).filter(
        Course.id == course_id, Course.user_id == current_user.id
    ).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    doc_count = db.query(Document).filter(Document.course_id == course.id).count()
    return CourseResponse(
        id=course.id,
        name=course.name,
        description=course.description,
        user_id=course.user_id,
        created_at=course.created_at,
        document_count=doc_count,
    )


@router.delete("/{course_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_course(
    course_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    course = db.query(Course).filter(
        Course.id == course_id, Course.user_id == current_user.id
    ).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    db.delete(course)
    db.commit()
