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
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    raise PDFExtractionError("This PDF is password-protected.")
            if len(reader.pages) > max_pages:
                raise PDFExtractionError(f"PDF has more than {max_pages} pages.")
            parts = []
            for page in reader.pages:
                try:
                    extracted = page.extract_text()
                except Exception:
                    continue  # skip a single broken page
                if extracted:
                    parts.append(extracted)
        except PDFExtractionError:
            raise
        except (PdfReadError, ValueError, KeyError, TypeError) as e:
            logger.warning("PDF parse failed: %s", e)
            raise PDFExtractionError("This file could not be read as a PDF.")
        text = "\n".join(parts).replace("\x00", "")
        if len(text.strip()) < 20:
            raise PDFExtractionError(
                "No text found in this PDF. Scanned/image-only PDFs are not supported yet."
            )
        return text
