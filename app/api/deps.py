"""Shared route dependencies."""
from fastapi import Depends, HTTPException, Path
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Course, User
from app.core.security import get_current_user


def get_owned_course(
    course_id: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Course:
    """Return the course if it belongs to the current user, else 404 (never 403, to avoid leaking IDs)."""
    course = db.query(Course).filter(Course.id == course_id, Course.user_id == current_user.id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    return course
