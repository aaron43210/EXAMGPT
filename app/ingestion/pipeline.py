"""
Background ingestion:
  PDF bytes -> pages -> chunks (+ page numbers) -> embeddings -> chunks table (+ pgvector)
            -> knowledge graph (syllabus / PYQ documents only)
Runs after the upload request has returned, with its own DB session.
"""
import logging
from sqlalchemy.orm import Session
from app.agents.llm_setup import LLMError
from app.db.database import SessionLocal
from app.db.models import Chunk, Document, DocumentStatus
from app.core.config import get_settings
from app.ingestion.chunker import Chunker
from app.ingestion.pdf_loader import PDFLoader, PDFExtractionError
from app.ingestion import embedder
from app.knowledge import graph
from app.retrieval import answer_cache
from app.retrieval.hybrid_retriever import store_vectors

logger = logging.getLogger(__name__)

MAX_STORED_TEXT_CHARS = 2_000_000
EMBED_BATCH = 64


def index_pages(db: Session, doc: Document, pages: list[tuple[int, str]]) -> int:
    """Chunk, embed and store a document's pages. Shared by uploads and the eval harness."""
    pieces = Chunker().chunk_pages(pages)
    texts = [p for _, p in pieces]
    vectors: list | None = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = embedder.embed_texts(texts[i:i + EMBED_BATCH])
        if batch is None:
            vectors = None
            break
        vectors.extend(batch)

    db.query(Chunk).filter(Chunk.document_id == doc.id).delete()
    rows = [
        Chunk(document_id=doc.id, course_id=doc.course_id, doc_type=doc.doc_type, source=doc.filename,
              chunk_index=i, page=page, content=content, embedding=vectors[i] if vectors else None)
        for i, (page, content) in enumerate(pieces)
    ]
    db.add_all(rows)
    db.flush()
    if vectors:
        store_vectors(db, [(r.id, r.embedding) for r in rows])
    doc.content_text = "\n".join(t for _, t in pages)[:MAX_STORED_TEXT_CHARS]
    doc.chunk_count = len(rows)
    answer_cache.invalidate(db, doc.course_id)  # cached answers may now be incomplete
    return len(rows)


def graph_applicable(doc: Document) -> bool:
    return get_settings().GRAPH_RAG_ENABLED and doc.doc_type in ("syllabus", "pyq") and bool(doc.content_text)


def build_graph(db: Session, doc: Document) -> None:
    """Extract the knowledge graph for syllabus/PYQ documents. Never fails the upload."""
    if not graph_applicable(doc):
        doc.kg_status = "skipped"
        return
    try:
        with db.begin_nested():
            graph.clear_document(db, doc.id)
            if doc.doc_type == "syllabus":
                n = graph.build_from_syllabus(db, doc.course_id, doc.id, doc.content_text)
            else:
                n = graph.build_from_pyq(db, doc.course_id, doc.id, doc.content_text)
        doc.kg_status = "ready"
        answer_cache.invalidate(db, doc.course_id)
        logger.info("Knowledge graph for document %s: %d items", doc.id, n)
    except (LLMError, ValueError) as e:
        logger.warning("Knowledge graph build failed for document %s: %s", doc.id, e)
        doc.kg_status = "failed"


def process_document(document_id: int, data: bytes) -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        if doc is None:
            return
        try:
            pages = PDFLoader.extract_pages(data, max_pages=settings.MAX_PDF_PAGES)
            n = index_pages(db, doc, pages)
            doc.status = DocumentStatus.ready.value
            doc.error_message = None
            doc.kg_status = "processing" if graph_applicable(doc) else "skipped"
            db.commit()
            logger.info("Indexed document %s: %d chunks", doc.id, n)
        except PDFExtractionError as e:
            db.rollback()
            doc.status = DocumentStatus.failed.value
            doc.error_message = str(e)
            db.commit()
            return
        except Exception:
            logger.exception("Ingestion failed for document %s", document_id)
            db.rollback()
            doc.status = DocumentStatus.failed.value
            doc.error_message = "Processing failed. Please try uploading again."
            db.commit()
            return
        build_graph(db, doc)
        db.commit()
    finally:
        db.close()


def rebuild_graph(document_id: int) -> None:
    db = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        if doc is not None:
            build_graph(db, doc)
            db.commit()
    finally:
        db.close()
