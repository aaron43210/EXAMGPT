"""
Study Plan API routes — generate and retrieve study plans for a course.
Retrieves document text directly from storage so study plans work even after server restarts.
"""
import json
import os
import tempfile
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Course, StudyPlan, Document, User
from app.schemas.planner import StudyPlanRequest, StudyPlanResponse
from app.core.security import get_current_user
from app.agents.llm_setup import get_llm
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.chunker import Chunker

logger = logging.getLogger(__name__)
router = APIRouter()

STUDY_PLAN_PROMPT = """You are ExamGPT, an expert academic study planner. Create a highly detailed, personalized study plan based on the ACTUAL course content extracted from the student's documents.

Course: {course_name}
Days until exam: {days}
Hours per day: {hours_per_day}
Total study hours: {total_hours}

## Actual Course Content (extracted from uploaded documents)

### Syllabus / Course Outline
{syllabus_text}

### Notes & Study Material
{notes_text}

### Previous Year Questions
{pyq_text}

## Instructions
Based on the REAL topics and content above, generate a comprehensive, specific study plan. Use the actual topic names from the documents — do NOT use generic placeholders like "topic 1" or "review all materials".

Generate a study plan as a JSON object:
{{
  "title": "Study Plan for {course_name}",
  "total_days": {days},
  "hours_per_day": {hours_per_day},
  "phases": [
    {{
      "phase": 1,
      "name": "Descriptive phase name based on real topics",
      "days": "Day 1 - Day X",
      "focus": "Specific topic from the syllabus",
      "topics": ["Real topic A", "Real topic B"],
      "activities": ["Read notes on [specific real topic]", "Solve PYQs from [year] on [topic]"],
      "hours": 15
    }}
  ],
  "tips": ["Tip specific to this course's content", "Another specific tip"],
  "priority_topics": ["Most important real topic 1", "Real topic 2"]
}}

Respond ONLY with the JSON, no other text."""


def _extract_text_from_doc(db_doc: Document) -> str:
    """Download from Supabase or read from local path, extract text."""
    text = ""
    try:
        # Try Supabase download first
        if db_doc.file_url and not db_doc.file_url.startswith("/"):
            from app.core.supabase_client import download_file
            # file_url is like "bucket/courses/1/file.pdf"
            parts = db_doc.file_url.split("/", 1)
            object_name = parts[1] if len(parts) > 1 else db_doc.file_url
            file_bytes = download_file(object_name)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            text = PDFLoader.extract_text(tmp_path)
            os.unlink(tmp_path)
        elif db_doc.file_url and os.path.exists(db_doc.file_url):
            # Local file path
            text = PDFLoader.extract_text(db_doc.file_url)
    except Exception as e:
        logger.warning(f"Could not extract text from doc {db_doc.id}: {e}")
    return text[:4000]  # Limit to avoid token overflow


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

    # Fetch ALL documents for this course
    all_docs = db.query(Document).filter(
        Document.course_id == course_id,
        Document.status == "ready",
    ).all()

    syllabus_texts, notes_texts, pyq_texts = [], [], []

    for db_doc in all_docs:
        text = _extract_text_from_doc(db_doc)
        if not text:
            continue
        if db_doc.doc_type == "syllabus":
            syllabus_texts.append(f"[{db_doc.filename}]\n{text}")
        elif db_doc.doc_type == "notes":
            notes_texts.append(f"[{db_doc.filename}]\n{text}")
        elif db_doc.doc_type == "pyq":
            pyq_texts.append(f"[{db_doc.filename}]\n{text}")

    syllabus_text = "\n\n---\n\n".join(syllabus_texts) if syllabus_texts else "No syllabus uploaded."
    notes_text = "\n\n---\n\n".join(notes_texts) if notes_texts else "No notes uploaded."
    pyq_text = "\n\n---\n\n".join(pyq_texts) if pyq_texts else "No PYQs uploaded."

    # If no documents found, try hybrid retriever as fallback
    if not syllabus_texts and not notes_texts:
        from app.retrieval.hybrid_retriever import hybrid_retriever
        syllabus_docs = hybrid_retriever.retrieve(
            query="syllabus topics units chapters overview",
            top_k=5, doc_type_filter="syllabus", course_id_filter=course_id,
        )
        notes_docs = hybrid_retriever.retrieve(
            query="main topics concepts definitions",
            top_k=5, doc_type_filter="notes", course_id_filter=course_id,
        )
        syllabus_text = "\n\n".join([d.page_content for d in syllabus_docs]) or "No syllabus available."
        notes_text = "\n\n".join([d.page_content for d in notes_docs]) or "No notes available."

    llm = get_llm()
    total_hours = request.days * request.hours_per_day

    prompt = STUDY_PLAN_PROMPT.format(
        course_name=course.name,
        days=request.days,
        hours_per_day=request.hours_per_day,
        total_hours=total_hours,
        syllabus_text=syllabus_text[:3000],
        notes_text=notes_text[:3000],
        pyq_text=pyq_text[:1500],
    )

    response = llm.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)

    # Strip markdown code fences if LLM wraps the JSON
    response_text = response_text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]

    try:
        plan_data = json.loads(response_text.strip())
    except (json.JSONDecodeError, AttributeError):
        logger.warning("JSON parse failed for study plan, using fallback.")
        plan_data = {
            "title": f"Study Plan for {course.name}",
            "total_days": request.days,
            "hours_per_day": request.hours_per_day,
            "phases": [{"phase": 1, "name": "Full Course Review", "days": f"Day 1 - Day {request.days}", "focus": "Review all uploaded materials", "topics": [], "activities": ["Review notes", "Practice PYQs"], "hours": total_hours}],
            "tips": ["Upload your syllabus for a more detailed plan", "Practice PYQs regularly"],
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
