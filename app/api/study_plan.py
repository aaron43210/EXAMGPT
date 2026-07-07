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

logger = logging.getLogger(__name__)
router = APIRouter()

STUDY_PLAN_PROMPT = """You are an expert academic study planner. Analyze the course content below and create a DAY-BY-DAY study schedule.

Course: {course_name}
Days until exam: {days}
Hours per day: {hours_per_day}
Total study hours: {total_hours}

## Actual Course Content (from student's uploaded documents)

### Syllabus / Course Outline
{syllabus_text}

### Notes & Study Material
{notes_text}

### Previous Year Questions
{pyq_text}

## IMPORTANT RULES
1. Extract EVERY topic and subtopic from the documents above.
2. Create a schedule where each day has specific topics assigned.
3. Each day MUST list the exact topics to study with time allocation.
4. For longer plans (>7 days), group into phases but still show daily breakdown.
5. Include revision days and PYQ practice days near the end.
6. Prioritize topics that appear in PYQs.

Generate a JSON object with this EXACT structure:
{{
  "title": "Study Plan for {course_name}",
  "total_days": {days},
  "hours_per_day": {hours_per_day},
  "schedule": [
    {{
      "day": 1,
      "date_label": "Day 1",
      "theme": "Introduction & Basics",
      "topics": [
        {{
          "name": "Exact topic name from document",
          "duration_hours": 1.5,
          "activity": "Read notes + make summary",
          "priority": "high"
        }},
        {{
          "name": "Another specific topic",
          "duration_hours": 2.0,
          "activity": "Study concepts + solve examples",
          "priority": "medium"
        }}
      ]
    }}
  ],
  "priority_topics": ["Most important topic 1", "Topic 2"],
  "tips": ["Tip 1", "Tip 2"]
}}

RESPOND ONLY WITH THE JSON. No markdown, no code fences, just raw JSON."""


def _extract_text_from_doc(db_doc: Document) -> str:
    """Download from Supabase or read from local path, extract text."""
    text = ""
    try:
        if db_doc.file_url and not db_doc.file_url.startswith("/"):
            from app.core.supabase_client import download_file
            # file_url could be 'bucket/courses/1/file.pdf' or just 'courses/1/file.pdf'
            if "courses/" in db_doc.file_url:
                object_name = "courses/" + db_doc.file_url.split("courses/", 1)[1]
            else:
                object_name = db_doc.file_url
            file_bytes = download_file(object_name)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            text = PDFLoader.extract_text(tmp_path)
            os.unlink(tmp_path)
        elif db_doc.file_url and os.path.exists(db_doc.file_url):
            text = PDFLoader.extract_text(db_doc.file_url)
    except Exception as e:
        logger.warning(f"Could not extract text from doc {db_doc.id}: {e}")
    return text[:4000]


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

    # Fallback to hybrid retriever if no documents found
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

    # Extract JSON block even if wrapped in text
    response_text = response_text.strip()
    import re
    match = re.search(r'```(?:json)?(.*?)```', response_text, re.DOTALL)
    if match:
        response_text = match.group(1).strip()
    else:
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1:
            response_text = response_text[start:end+1]

    try:
        plan_data = json.loads(response_text.strip())
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"JSON parse failed for study plan: {e}")
        logger.warning(f"Raw response (first 500 chars): {response_text[:500]}")
        plan_data = {
            "title": f"Study Plan for {course.name}",
            "total_days": request.days,
            "hours_per_day": request.hours_per_day,
            "schedule": [
                {
                    "day": d + 1,
                    "date_label": f"Day {d + 1}",
                    "theme": "Study Session",
                    "topics": [{"name": f"Error: No docs found or LLM parse failed (sys={len(syllabus_text)}, notes={len(notes_text)})", "duration_hours": request.hours_per_day, "activity": "Read and summarize", "priority": "medium"}],
                }
                for d in range(min(request.days, 7))
            ],
            "tips": [f"Debug: syllabus len: {len(syllabus_text)}, notes len: {len(notes_text)}", f"Error: {str(e)}", f"Raw: {response_text[:100]}", "Practice PYQs regularly"],
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
