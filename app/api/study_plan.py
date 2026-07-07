"""
Study Plan API routes — generate and retrieve study plans for a course.
"""
import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Course, StudyPlan, User
from app.schemas.planner import StudyPlanRequest, StudyPlanResponse
from app.core.security import get_current_user
from app.agents.llm_setup import get_llm
from app.retrieval.hybrid_retriever import hybrid_retriever

logger = logging.getLogger(__name__)
router = APIRouter()

STUDY_PLAN_PROMPT = """You are ExamGPT, a study planning expert. Generate a structured, personalized study plan based on the actual course content provided below.

Course: {course_name}
Available days: {days}
Hours per day: {hours_per_day}
Total available hours: {total_hours}

## Real Course Content (extracted from the student's uploaded documents)
Use the following actual content to identify real topics, units, and subjects from this course:

### Syllabus Excerpts
{syllabus_context}

### Notes Excerpts
{notes_context}

Based on the above real course content, generate a study plan as a JSON object with this structure:
{{
  "title": "Study Plan for [course name]",
  "total_days": {days},
  "hours_per_day": {hours_per_day},
  "phases": [
    {{
      "phase": 1,
      "name": "Phase name based on actual topics",
      "days": "Day X - Day Y",
      "focus": "What specific topics to focus on",
      "topics": ["real topic 1 from documents", "real topic 2"],
      "activities": ["Read notes on [specific topic]", "Practice PYQs for [specific topic]"],
      "hours": 10
    }}
  ],
  "tips": ["Specific study tip based on the syllabus", "Study tip 2"],
  "priority_topics": ["high priority topic from syllabus 1", "topic 2"]
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

    # Retrieve actual document content using hybrid retriever
    syllabus_docs = hybrid_retriever.retrieve(
        query="syllabus topics units chapters overview",
        top_k=5,
        doc_type_filter="syllabus",
        course_id_filter=course_id,
    )
    notes_docs = hybrid_retriever.retrieve(
        query="main topics concepts definitions",
        top_k=5,
        doc_type_filter="notes",
        course_id_filter=course_id,
    )

    syllabus_context = "\n\n".join([
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in syllabus_docs
    ]) if syllabus_docs else "No syllabus documents uploaded yet."

    notes_context = "\n\n".join([
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in notes_docs
    ]) if notes_docs else "No notes documents uploaded yet."

    llm = get_llm()
    total_hours = request.days * request.hours_per_day

    prompt = STUDY_PLAN_PROMPT.format(
        course_name=course.name,
        days=request.days,
        hours_per_day=request.hours_per_day,
        total_hours=total_hours,
        syllabus_context=syllabus_context,
        notes_context=notes_context,
    )

    response = llm.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)

    try:
        plan_data = json.loads(response_text.strip())
    except (json.JSONDecodeError, AttributeError):
        plan_data = {
            "title": f"Study Plan for {course.name}",
            "total_days": request.days,
            "hours_per_day": request.hours_per_day,
            "phases": [{"phase": 1, "name": "General Study", "days": f"Day 1 - Day {request.days}", "focus": "Review all materials", "topics": [], "activities": ["Review notes", "Practice PYQs"], "hours": total_hours}],
            "tips": ["Start with syllabus topics", "Practice PYQs regularly"],
            "priority_topics": [],
        }

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
