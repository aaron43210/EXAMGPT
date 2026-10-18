import io
import logging
import pypdf
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised with a user-safe message when a PDF can't be processed."""


class PDFLoader:
    @staticmethod
    def extract_text(data: bytes, max_pages: int = 500) -> str:
        return "\n".join(text for _, text in PDFLoader.extract_pages(data, max_pages))

    @staticmethod
    def extract_pages(data: bytes, max_pages: int = 500) -> list[tuple[int, str]]:
        """Return [(page_number, text)] for pages that contain text (1-based)."""
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    raise PDFExtractionError("This PDF is password-protected.")
            if len(reader.pages) > max_pages:
                raise PDFExtractionError(f"PDF has more than {max_pages} pages.")
            pages = []
            for number, page in enumerate(reader.pages, 1):
                try:
                    extracted = page.extract_text()
                except Exception:
                    continue  # skip a single broken page
                if extracted and extracted.strip():
                    pages.append((number, extracted.replace("\x00", "")))
        except PDFExtractionError:
            raise
        except (PdfReadError, ValueError, KeyError, TypeError) as e:
            logger.warning("PDF parse failed: %s", e)
            raise PDFExtractionError("This file could not be read as a PDF.")
        if sum(len(t.strip()) for _, t in pages) < 20:
            raise PDFExtractionError(
                "No text found in this PDF. Scanned/image-only PDFs are not supported yet."
            )
        return pages
