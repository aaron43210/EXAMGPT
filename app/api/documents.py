"""
Document API routes — upload, list and delete documents for a course.
Uploads are validated, stored, and indexed in a background task.
"""
import os
import re
import unicodedata
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Form, Path, status
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Course, Chunk, Document, PYQQuestion, User, DocumentStatus, DocumentType
from app.schemas.document import DocumentResponse, DocumentListResponse
from app.core.security import get_current_user
from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.api.deps import get_owned_course
from app.ingestion.pipeline import process_document

router = APIRouter()

_READ_CHUNK = 1024 * 1024


def safe_filename(name: str | None) -> str:
    name = unicodedata.normalize("NFKC", os.path.basename((name or "").replace("\\", "/")))
    name = re.sub(r"[^\w.\- ()]", "_", name).strip(" ._") or "document.pdf"
    stem, ext = os.path.splitext(name)
    return f"{stem[:120]}{ext.lower()}"


def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    buf = bytearray()
    while True:
        part = file.file.read(_READ_CHUNK)
        if not part:
            return bytes(buf)
        buf.extend(part)
        if len(buf) > max_bytes:
            raise HTTPException(status_code=413, detail=f"File exceeds {max_bytes // (1024 * 1024)}MB limit")


@router.post("/{course_id}/documents", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: str = Form(default="notes"),
    course: Course = Depends(get_owned_course),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    settings = get_settings()
    limiter.check(f"upload:{current_user.id}", settings.RATE_LIMIT_UPLOAD_PER_MIN)

    if doc_type not in {dt.value for dt in DocumentType}:
        raise HTTPException(status_code=400, detail="doc_type must be one of: syllabus, notes, pyq")

    filename = safe_filename(file.filename)
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if db.query(Document).filter(Document.course_id == course.id).count() >= settings.MAX_DOCUMENTS_PER_COURSE:
        raise HTTPException(status_code=400,
                            detail=f"A course can have at most {settings.MAX_DOCUMENTS_PER_COURSE} documents")

    data = _read_limited(file, settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024)
    if not data:
        raise HTTPException(status_code=400, detail="The file is empty")
    if not data[:1024].lstrip().startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="This file is not a valid PDF")

    db_doc = Document(
        title=filename,
        filename=filename,
        doc_type=doc_type,
        status=DocumentStatus.processing.value,
        course_id=course.id,
        user_id=current_user.id,
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    background_tasks.add_task(process_document, db_doc.id, data)
    return db_doc


@router.get("/{course_id}/documents", response_model=DocumentListResponse)
def list_documents(course: Course = Depends(get_owned_course), db: Session = Depends(get_db)):
    docs = db.query(Document).filter(Document.course_id == course.id).order_by(Document.created_at.desc()).all()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete("/{course_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: int = Path(..., ge=1),
    course: Course = Depends(get_owned_course),
    db: Session = Depends(get_db),
):
    doc = db.query(Document).filter(Document.id == document_id, Document.course_id == course.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    db.query(Chunk).filter(Chunk.document_id == doc.id).delete(synchronize_session=False)
    db.query(PYQQuestion).filter(PYQQuestion.document_id == doc.id).delete(synchronize_session=False)
    db.delete(doc)
    db.commit()
