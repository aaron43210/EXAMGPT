"""
Course-scoped hybrid retrieval — the retrieval stage of the advanced RAG pipeline.

  1. Sparse:  BM25 (Lucene IDF) over an in-memory inverted index per (course, doc type).
              The index is cached and rebuilt automatically when the course's chunks change.
  2. Dense:   embedding cosine similarity — pgvector HNSW index on Postgres when available,
              otherwise an in-memory matrix of the stored vectors.
  3. Fusion:  Reciprocal Rank Fusion (RRF) of the sparse and dense rankings.
  4. Rerank:  cross-encoder over the fused top candidates.

Every lookup is filtered by course_id, so users only ever see their own course.
"""
import logging
import re
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass
import numpy as np
from sqlalchemy import func, text
from sqlalchemy.orm import Session, defer
from app.core.config import get_settings
from app.db.models import Chunk
from app.ingestion import embedder
from app.retrieval import reranker

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "or", "what", "how", "why", "explain",
         "describe", "define", "with", "for", "on", "it", "this", "that", "me", "about", "do", "does"}
_RRF_K = 60
RERANK_POOL = 20
DENSE_CANDIDATES = 100
VECTOR_DIM = 384  # matches the pgvector column created by the migration
INDEX_CACHE_SIZE = 64


@dataclass
class Hit:
    chunk: Chunk
    score: float          # final score used for ordering
    rerank: float | None  # cross-encoder probability, when reranked


@dataclass
class _Index:
    """Inverted index (+ optional vector matrix) over one course's chunks of one doc type."""
    version: tuple
    ids: list[int]
    pos: dict[int, int]
    lengths: np.ndarray
    avgdl: float
    postings: dict[str, tuple[np.ndarray, np.ndarray]]  # term -> (row positions, term frequencies)
    has_vectors_loaded: bool
    vectors: np.ndarray | None = None      # (rows_with_vectors, dim), L2-normalized
    vector_rows: np.ndarray | None = None  # row positions of `vectors`


_cache: "OrderedDict[tuple, _Index]" = OrderedDict()
_cache_lock = threading.Lock()
_pgvector_cache: dict[str, bool] = {}


def reset_index_cache() -> None:
    """Drop cached indexes (tests and the eval harness recreate databases with reused ids)."""
    with _cache_lock:
        _cache.clear()
    _pgvector_cache.clear()


def _tokenize(value: str) -> list[str]:
    return _TOKEN.findall(value.lower())


def content_terms(value: str) -> set[str]:
    return {t for t in _tokenize(value) if t not in _STOP and len(t) > 1}


# ── pgvector ────────────────────────────────────────────────────────────

def pgvector_enabled(db: Session) -> bool:
    bind = db.get_bind()
    if bind.dialect.name != "postgresql":
        return False
    key = str(bind.url)
    if key not in _pgvector_cache:
        _pgvector_cache[key] = db.execute(text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'embedding_vec'")).first() is not None
    return _pgvector_cache[key]


def store_vectors(db: Session, rows: list[tuple[int, list[float]]]) -> None:
    """Mirror embeddings into the pgvector column (no-op elsewhere)."""
    if not rows or not pgvector_enabled(db):
        return
    usable = [(cid, vec) for cid, vec in rows if len(vec) == VECTOR_DIM]
    if len(usable) < len(rows):
        logger.warning("Skipping %d vectors whose dimension != %d (EMBEDDING_MODEL changed?)",
                       len(rows) - len(usable), VECTOR_DIM)
    if usable:
        db.execute(text("UPDATE chunks SET embedding_vec = CAST(:v AS vector) WHERE id = :id"),
                   [{"id": cid, "v": str(vec)} for cid, vec in usable])


# ── Index cache ─────────────────────────────────────────────────────────

def _version(db: Session, course_id: int, doc_type: str | None) -> tuple:
    q = db.query(func.count(Chunk.id), func.max(Chunk.id)).filter(Chunk.course_id == course_id)
    if doc_type:
        q = q.filter(Chunk.doc_type == doc_type)
    return tuple(q.one())


def _build_index(db: Session, course_id: int, doc_type: str | None, version: tuple,
                 load_vectors: bool) -> _Index:
    cols = [Chunk.id, Chunk.content] + ([Chunk.embedding] if load_vectors else [])
    q = db.query(*cols).filter(Chunk.course_id == course_id)
    if doc_type:
        q = q.filter(Chunk.doc_type == doc_type)
    ids, lengths, vec_rows, vecs = [], [], [], []
    postings: dict[str, tuple[list, list]] = {}
    for i, row in enumerate(q.order_by(Chunk.id)):
        ids.append(row.id)
        tokens = _tokenize(row.content)
        lengths.append(len(tokens))
        for term, tf in Counter(tokens).items():
            rows, tfs = postings.setdefault(term, ([], []))
            rows.append(i)
            tfs.append(tf)
        if load_vectors and row.embedding:
            vec_rows.append(i)
            vecs.append(row.embedding)
    lengths_arr = np.array(lengths, dtype=np.float64)
    index = _Index(
        version=version,
        ids=ids,
        pos={cid: i for i, cid in enumerate(ids)},
        lengths=lengths_arr,
        avgdl=float(lengths_arr.sum() / len(ids)) if ids and lengths_arr.sum() else 1.0,
        postings={t: (np.array(r, dtype=np.int64), np.array(f, dtype=np.float64)) for t, (r, f) in postings.items()},
        has_vectors_loaded=load_vectors,
    )
    if vecs:
        index.vectors = np.array(vecs, dtype=np.float32)
        index.vector_rows = np.array(vec_rows, dtype=np.int64)
    return index


def _get_index(db: Session, course_id: int, doc_type: str | None, load_vectors: bool) -> _Index:
    key = (course_id, doc_type or "*")
    version = _version(db, course_id, doc_type)
    with _cache_lock:
        index = _cache.get(key)
        if index is not None and index.version == version and (index.has_vectors_loaded or not load_vectors):
            _cache.move_to_end(key)
            return index
    index = _build_index(db, course_id, doc_type, version, load_vectors)
    with _cache_lock:
        _cache[key] = index
        _cache.move_to_end(key)
        while len(_cache) > INDEX_CACHE_SIZE:
            _cache.popitem(last=False)
    return index


# ── Scoring ─────────────────────────────────────────────────────────────

def bm25_scores(index: _Index, query_tokens: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Okapi BM25 with Lucene's always-positive IDF (works for 1-document corpora)."""
    n = len(index.ids)
    scores = np.zeros(n)
    if n == 0:
        return scores
    norm = k1 * (1 - b + b * index.lengths / index.avgdl)
    for term in set(query_tokens):
        posting = index.postings.get(term)
        if posting is None:
            continue
        rows, tf = posting
        df = len(rows)
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        scores[rows] += idf * tf * (k1 + 1) / (tf + norm[rows])
    return scores


def _dense_order(db: Session, index: _Index, query: str, course_id: int, doc_type: str | None,
                 use_pgvector: bool) -> list[int]:
    """Row positions in `index`, most similar first."""
    if use_pgvector:
        qvec = embedder.embed_query(query)
        if qvec is None:
            return []
        sql = ("SELECT id FROM chunks WHERE course_id = :cid AND embedding_vec IS NOT NULL "
               + ("AND doc_type = :dt " if doc_type else "")
               + "ORDER BY embedding_vec <=> CAST(:q AS vector) LIMIT :lim")
        ids = [r[0] for r in db.execute(text(sql), {"cid": course_id, "dt": doc_type,
                                                    "q": str(qvec.tolist()), "lim": DENSE_CANDIDATES})]
        return [index.pos[i] for i in ids if i in index.pos]
    if index.vectors is None:
        return []
    qvec = embedder.embed_query(query)
    if qvec is None:
        return []
    sims = index.vectors @ qvec
    return [int(index.vector_rows[j]) for j in np.argsort(-sims)[:DENSE_CANDIDATES]]


def search(db: Session, course_id: int, query: str, top_k: int = 4,
           doc_type: str | None = None, use_rerank: bool = True) -> list[Hit]:
    if not query.strip():
        return []
    settings = get_settings()
    use_pgvector = pgvector_enabled(db)
    index = _get_index(db, course_id, doc_type, load_vectors=settings.EMBEDDINGS_ENABLED and not use_pgvector)
    n = len(index.ids)
    if n == 0:
        return []

    fused = np.zeros(n)
    tokens = _tokenize(query)
    if tokens and settings.BM25_ENABLED:
        scores = bm25_scores(index, tokens)
        for rank, idx in enumerate(np.argsort(-scores)):
            if scores[idx] <= 0:
                break
            fused[idx] += 1.0 / (_RRF_K + rank)
    if settings.EMBEDDINGS_ENABLED:
        for rank, idx in enumerate(_dense_order(db, index, query, course_id, doc_type, use_pgvector)):
            fused[idx] += 1.0 / (_RRF_K + rank)

    order = [int(i) for i in np.argsort(-fused) if fused[i] > 0]
    if not order:
        return []

    rerank_on = use_rerank and settings.RERANKER_ENABLED
    pool = order[:RERANK_POOL] if rerank_on else order[:top_k]
    rows = {c.id: c for c in db.query(Chunk).options(defer(Chunk.embedding))
            .filter(Chunk.id.in_([index.ids[i] for i in pool]))}
    pool = [i for i in pool if index.ids[i] in rows]  # a document may be deleted mid-request

    if rerank_on:
        probs = reranker.rerank(query, [rows[index.ids[i]].content for i in pool])
        if probs is not None:
            ranked = sorted(zip(pool, probs), key=lambda p: -p[1])[:top_k]
            return [Hit(rows[index.ids[i]], p, p) for i, p in ranked]
    return [Hit(rows[index.ids[i]], float(fused[i]), None) for i in pool[:top_k]]


def confidence(query: str, hits: list[Hit]) -> float:
    """0-1 estimate of whether retrieval found relevant material for the query."""
    if not hits:
        return 0.0
    if hits[0].rerank is not None:
        return round(max(h.rerank for h in hits), 3)
    terms = content_terms(query)
    if not terms:
        return 0.5
    best = max(len(terms & content_terms(h.chunk.content)) / len(terms) for h in hits)
    return round(best, 3)
