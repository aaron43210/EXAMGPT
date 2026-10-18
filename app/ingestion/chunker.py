import re


class Chunker:
    """Splits text into overlapping chunks, preferring paragraph/line/word boundaries."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[str]:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        chunks, start, n = [], 0, len(text)
        while start < n:
            end = min(start + self.chunk_size, n)
            if end < n:
                window = text[start:end]
                for sep in ("\n\n", "\n", ". ", " "):
                    cut = window.rfind(sep)
                    if cut > self.chunk_size // 2:
                        end = start + cut + len(sep)
                        break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= n:
                break
            start = max(end - self.chunk_overlap, start + 1)
        return chunks
