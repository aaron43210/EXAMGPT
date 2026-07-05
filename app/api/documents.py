"""
Document API routes — upload and list documents for a course.
"""
import os
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Course, Document, User, DocumentStatus, DocumentType
from app.schemas.document import DocumentUploadResponse, DocumentResponse, DocumentListResponse
from app.core.security import get_current_user
from app.core.config import get_settings
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.chunker import Chunker
from app.retrieval.hybrid_retriever import hybrid_retriever
from langchain_core.documents import Document as LCDocument

router = APIRouter()


def _verify_course_owner(db: Session, course_id: int, user_id: int) -> Course:
    course = db.query(Course).filter(
        Course.id == course_id, Course.user_id == user_id
    ).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    return course


@router.post("/{course_id}/documents", response_model=DocumentUploadResponse)
def upload_document(
    course_id: int,
    file: UploadFile = File(...),
    doc_type: str = Form(default="notes"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    settings = get_settings()
    course = _verify_course_owner(db, course_id, current_user.id)

    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Validate file size
    file_data = file.file.read()
    if len(file_data) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit")

    # Validate doc_type
    if doc_type not in [dt.value for dt in DocumentType]:
        doc_type = DocumentType.notes.value

    # Save to DB with processing status
    db_doc = Document(
        title=file.filename,
        filename=file.filename,
        doc_type=doc_type,
        status=DocumentStatus.processing.value,
        course_id=course.id,
        user_id=current_user.id,
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    # Process synchronously (Celery can be added later)
    try:
        # Save file locally for processing
        upload_dir = os.path.join(settings.UPLOAD_DIR, str(course_id))
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file_data)

        # Try Supabase upload (graceful fallback if not available)
        try:
            from app.core.supabase_client import upload_file
            object_name = f"courses/{course_id}/{file.filename}"
            db_doc.file_url = upload_file(file_data, object_name)
        except Exception:
            db_doc.file_url = file_path

        # Extract text
        text = PDFLoader.extract_text(file_path)

        # Chunk and index
        chunker = Chunker()
        docs = chunker.chunk_text(text, metadata={
            "source": file.filename,
            "user_id": current_user.id,
            "course_id": course_id,
            "doc_type": doc_type,
        })
        hybrid_retriever.add_documents(docs)

        # Update status
        db_doc.status = DocumentStatus.ready.value
        db.commit()

    except Exception as e:
        db_doc.status = DocumentStatus.failed.value
        db.commit()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {repr(e)}")

    return DocumentUploadResponse(
        id=db_doc.id,
        filename=db_doc.filename,
        doc_type=db_doc.doc_type,
        status=db_doc.status,
        message="Upload successful and indexed.",
    )


@router.get("/{course_id}/documents", response_model=DocumentListResponse)
def list_documents(
    course_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _verify_course_owner(db, course_id, current_user.id)
    docs = db.query(Document).filter(Document.course_id == course_id).all()
    doc_responses = [
        DocumentResponse(
            id=d.id,
            title=d.title,
            filename=d.filename,
            doc_type=d.doc_type,
            status=d.status,
            course_id=d.course_id,
            created_at=d.created_at,
        )
        for d in docs
    ]
    return DocumentListResponse(documents=doc_responses, total=len(doc_responses))
