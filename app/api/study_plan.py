"""
Study Plan API routes — generate and retrieve study plans for a course.
"""
import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Course, StudyPlan, Document, User
from app.schemas.planner import StudyPlanRequest, StudyPlanResponse
from app.core.security import get_current_user
from app.agents.llm_setup import get_llm

logger = logging.getLogger(__name__)
router = APIRouter()

STUDY_PLAN_PROMPT = """You are ExamGPT, a study planning expert. Generate a structured study plan based on the following course information.

Course: {course_name}
Available days: {days}
Hours per day: {hours_per_day}
Total available hours: {total_hours}

Document types available:
- Syllabus documents: {syllabus_count}
- Notes documents: {notes_count}
- PYQ documents: {pyq_count}

Generate a study plan as a JSON object with this structure:
{{
  "title": "Study Plan for [course name]",
  "total_days": {days},
  "hours_per_day": {hours_per_day},
  "phases": [
    {{
      "phase": 1,
      "name": "Phase name",
      "days": "Day X - Day Y",
      "focus": "What to focus on",
      "topics": ["topic1", "topic2"],
      "activities": ["Read notes on X", "Practice PYQs for Y"],
      "hours": 10
    }}
  ],
  "tips": ["Study tip 1", "Study tip 2"],
  "priority_topics": ["high priority topic 1", "topic 2"]
}}

Respond ONLY with the JSON."""


@router.post("/{course_id}/study-plan", response_model=StudyPlanResponse)
def generate_study_plan(
    course_id: int,
    request: StudyPlanRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    course = db.query(Course).filter(
        Course.id == course_id, Course.user_id == current_user.id
    ).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Count documents by type
    syllabus_count = db.query(Document).filter(
        Document.course_id == course_id, Document.doc_type == "syllabus"
    ).count()
    notes_count = db.query(Document).filter(
        Document.course_id == course_id, Document.doc_type == "notes"
    ).count()
    pyq_count = db.query(Document).filter(
        Document.course_id == course_id, Document.doc_type == "pyq"
    ).count()

    llm = get_llm()
    total_hours = request.days * request.hours_per_day

    prompt = STUDY_PLAN_PROMPT.format(
        course_name=course.name,
        days=request.days,
        hours_per_day=request.hours_per_day,
        total_hours=total_hours,
        syllabus_count=syllabus_count,
        notes_count=notes_count,
        pyq_count=pyq_count,
    )

    response = llm.invoke(prompt)

    try:
        plan_data = json.loads(response.strip())
    except (json.JSONDecodeError, AttributeError):
        plan_data = {
            "title": f"Study Plan for {course.name}",
            "total_days": request.days,
            "hours_per_day": request.hours_per_day,
            "phases": [{"phase": 1, "name": "General Study", "days": f"Day 1 - Day {request.days}", "focus": "Review all materials", "topics": [], "activities": ["Review notes", "Practice PYQs"], "hours": total_hours}],
            "tips": ["Start with syllabus topics", "Practice PYQs regularly"],
            "priority_topics": [],
        }

    # Save or update
    existing = db.query(StudyPlan).filter(
        StudyPlan.course_id == course_id, StudyPlan.user_id == current_user.id
    ).first()

    if existing:
        existing.plan_data = plan_data
        db.commit()
        db.refresh(existing)
        plan = existing
    else:
        plan = StudyPlan(
            course_id=course_id,
            user_id=current_user.id,
            plan_data=plan_data,
        )
        db.add(plan)
        db.commit()
        db.refresh(plan)

    return StudyPlanResponse(
        id=plan.id,
        course_id=plan.course_id,
        plan_data=plan.plan_data,
        created_at=plan.created_at,
    )


@router.get("/{course_id}/study-plan", response_model=StudyPlanResponse)
def get_study_plan(
    course_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    course = db.query(Course).filter(
        Course.id == course_id, Course.user_id == current_user.id
    ).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    plan = db.query(StudyPlan).filter(
        StudyPlan.course_id == course_id, StudyPlan.user_id == current_user.id
    ).order_by(StudyPlan.created_at.desc()).first()

    if not plan:
        raise HTTPException(status_code=404, detail="No study plan found. Generate one first.")

    return StudyPlanResponse(
        id=plan.id,
        course_id=plan.course_id,
        plan_data=plan.plan_data,
        created_at=plan.created_at,
    )
