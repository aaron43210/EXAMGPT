"""
Background ingestion: PDF bytes -> text -> chunks (+ embeddings) -> database.
Runs after the upload request has returned, with its own DB session.
"""
import logging
from app.db.database import SessionLocal
from app.db.models import Chunk, Document, DocumentStatus
from app.core.config import get_settings
from app.ingestion.chunker import Chunker
from app.ingestion.pdf_loader import PDFLoader, PDFExtractionError
from app.ingestion import embedder

logger = logging.getLogger(__name__)

MAX_STORED_TEXT_CHARS = 2_000_000
EMBED_BATCH = 64


def process_document(document_id: int, data: bytes) -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        if doc is None:
            return
        try:
            text = PDFLoader.extract_text(data, max_pages=settings.MAX_PDF_PAGES)[:MAX_STORED_TEXT_CHARS]
            pieces = Chunker().chunk_text(text)
            vectors: list | None = []
            for i in range(0, len(pieces), EMBED_BATCH):
                batch = embedder.embed_texts(pieces[i:i + EMBED_BATCH])
                if batch is None:
                    vectors = None
                    break
                vectors.extend(batch)

            db.query(Chunk).filter(Chunk.document_id == doc.id).delete()
            db.add_all([
                Chunk(
                    document_id=doc.id,
                    course_id=doc.course_id,
                    doc_type=doc.doc_type,
                    source=doc.filename,
                    chunk_index=i,
                    content=piece,
                    embedding=vectors[i] if vectors else None,
                )
                for i, piece in enumerate(pieces)
            ])
            doc.content_text = text
            doc.chunk_count = len(pieces)
            doc.status = DocumentStatus.ready.value
            doc.error_message = None
            db.commit()
            logger.info("Indexed document %s: %d chunks", doc.id, len(pieces))
        except PDFExtractionError as e:
            db.rollback()
            doc.status = DocumentStatus.failed.value
            doc.error_message = str(e)
            db.commit()
        except Exception:
            logger.exception("Ingestion failed for document %s", document_id)
            db.rollback()
            doc.status = DocumentStatus.failed.value
            doc.error_message = "Processing failed. Please try uploading again."
            db.commit()
    finally:
        db.close()
