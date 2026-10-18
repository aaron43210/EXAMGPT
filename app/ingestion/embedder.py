"""
Local embeddings via fastembed (ONNX, CPU). No external API needed.
If the model can't load, retrieval falls back to BM25 only.
"""
import logging
import os
import threading
import numpy as np
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_model = None
_failed = False
_lock = threading.Lock()


def _get_model():
    global _model, _failed
    settings = get_settings()
    if not settings.EMBEDDINGS_ENABLED or _failed:
        return None
    if _model is None:
        with _lock:
            if _model is None and not _failed:
                try:
                    from fastembed import TextEmbedding
                    _model = TextEmbedding(model_name=settings.EMBEDDING_MODEL,
                                           cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))
                except Exception as e:
                    logger.warning("Embedding model unavailable, using keyword search only: %s", e)
                    _failed = True
    return _model


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    model = _get_model()
    if model is None or not texts:
        return None
    vectors = np.array(list(model.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).round(6).tolist()


def embed_query(text: str) -> np.ndarray | None:
    vecs = embed_texts([text])
    return np.array(vecs[0], dtype=np.float32) if vecs else None


def warmup() -> None:
    embed_texts(["warmup"])
