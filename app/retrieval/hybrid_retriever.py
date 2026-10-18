"""
Course-scoped hybrid retrieval over chunks stored in the database.
BM25 keyword scores and embedding cosine scores are fused with Reciprocal Rank Fusion.
Every query is filtered by course_id in SQL, so users only ever see their own course.
"""
import re
import numpy as np
from collections import Counter
from sqlalchemy.orm import Session
from app.db.models import Chunk
from app.ingestion import embedder

_TOKEN = re.compile(r"[a-z0-9]+")
_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _bm25_scores(docs: list[list[str]], query: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Okapi BM25 with Lucene's always-positive IDF (works for 1-document corpora)."""
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df = Counter(t for d in docs for t in set(d))
    scores = np.zeros(n)
    for i, d in enumerate(docs):
        tf = Counter(d)
        norm = k1 * (1 - b + b * len(d) / avgdl)
        for t in set(query):
            f = tf.get(t)
            if f:
                idf = np.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                scores[i] += idf * f * (k1 + 1) / (f + norm)
    return scores


def retrieve(db: Session, course_id: int, query: str, top_k: int = 4,
             doc_type: str | None = None) -> list[Chunk]:
    q = db.query(Chunk).filter(Chunk.course_id == course_id)
    if doc_type:
        q = q.filter(Chunk.doc_type == doc_type)
    chunks = q.all()
    if not chunks or not query.strip():
        return []

    fused = np.zeros(len(chunks))

    query_tokens = _tokenize(query)
    if query_tokens:
        scores = _bm25_scores([_tokenize(c.content) for c in chunks], query_tokens)
        for rank, idx in enumerate(np.argsort(-scores)):
            if scores[idx] <= 0:
                break
            fused[idx] += 1.0 / (_RRF_K + rank)

    with_vec = [i for i, c in enumerate(chunks) if c.embedding]
    if with_vec:
        qvec = embedder.embed_query(query)
        if qvec is not None:
            matrix = np.array([chunks[i].embedding for i in with_vec], dtype=np.float32)
            sims = matrix @ qvec
            for rank, j in enumerate(np.argsort(-sims)):
                fused[with_vec[j]] += 1.0 / (_RRF_K + rank)

    order = [i for i in np.argsort(-fused) if fused[i] > 0][:top_k]
    return [chunks[i] for i in order]
