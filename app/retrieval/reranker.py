"""
Cross-encoder reranker (fastembed, ONNX on CPU). Scores (query, passage) pairs jointly,
which is far more precise than embedding similarity for picking the final top-k.
"""
import logging
import math
import os
import threading
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_model = None
_failed = False
_lock = threading.Lock()


def _get_model():
    global _model, _failed
    settings = get_settings()
    if not settings.RERANKER_ENABLED or _failed:
        return None
    if _model is None:
        with _lock:
            if _model is None and not _failed:
                try:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder
                    _model = TextCrossEncoder(model_name=settings.RERANKER_MODEL,
                                              cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))
                except Exception as e:
                    logger.warning("Reranker unavailable, skipping rerank: %s", e)
                    _failed = True
    return _model


def rerank(query: str, passages: list[str]) -> list[float] | None:
    """Return a relevance probability (0-1) per passage, or None if the reranker is off."""
    model = _get_model()
    if model is None or not passages:
        return None
    return [1 / (1 + math.exp(-float(s))) for s in model.rerank(query, passages)]


def warmup() -> None:
    rerank("warmup", ["warmup"])
