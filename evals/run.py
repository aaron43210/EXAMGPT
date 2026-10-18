"""
ExamGPT offline evaluation harness.

  python -m evals.run --suite routing                    # deterministic, no API key
  python -m evals.run --suite retrieval --config all     # deterministic ablation: bm25 / dense / hybrid / hybrid+rerank
  python -m evals.run --suite generation --limit 10      # needs GROQ_API_KEY (answer + LLM judge)
  python -m evals.run --suite safety                     # needs GROQ_API_KEY
  python -m evals.run --suite graph                      # needs GROQ_API_KEY (knowledge-graph extraction)
  python -m evals.run --suite retrieval --check          # fail (exit 1) on regression vs evals/baselines
  python -m evals.run --suite retrieval --update-baseline

Runs against a throwaway SQLite database — never your real DATABASE_URL.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_tmp = tempfile.mkdtemp(prefix="examgpt-eval-")
os.environ["DATABASE_URL"] = os.environ.get("EVAL_DATABASE_URL", f"sqlite:///{_tmp}/eval.db")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("SECRET_KEY", "eval-only-secret-key-not-used-for-anything-real")
os.environ["ONLINE_EVAL_SAMPLE_RATE"] = "0"

from app.core.config import get_settings  # noqa: E402
from app.core.stats import bootstrap_ci, mean, percentile  # noqa: E402
from evals import metrics as M  # noqa: E402

CONFIGS = {
    "bm25": {"embeddings": False, "rerank": False, "sparse": True},
    "dense": {"embeddings": True, "rerank": False, "sparse": False},
    "hybrid": {"embeddings": True, "rerank": False, "sparse": True},
    "hybrid_rerank": {"embeddings": True, "rerank": True, "sparse": True},
}
HIGHER_IS_BETTER_TOLERANCE = 0.02
DOCS = [("syllabus.txt", "syllabus.pdf", "syllabus"), ("notes.txt", "notes.pdf", "notes"),
        ("pyq.txt", "pyq.pdf", "pyq"), ("injected.txt", "interrupts.pdf", "notes")]


# ── Dataset ─────────────────────────────────────────────────────────────

def read_pages(path: Path) -> list[tuple[int, str]]:
    pages, current, number = [], [], 1
    for line in path.read_text().splitlines():
        if line.startswith("=== page "):
            if current:
                pages.append((number, "\n".join(current)))
            number, current = int(line.split()[2]), []
        else:
            current.append(line)
    if current:
        pages.append((number, "\n".join(current)))
    return pages


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _add_vector_index(engine) -> None:
    """On Postgres with pgvector (EVAL_DATABASE_URL), mirror the production migration's vector index."""
    from sqlalchemy import text
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as conn:
        if conn.execute(text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")).first() is None:
            return
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_vec vector(384)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw ON chunks "
                          "USING hnsw (embedding_vec vector_cosine_ops)"))


def setup_course(dataset: Path, build_graph: bool = False):
    from app.db.database import Base, SessionLocal, engine
    from app.db.models import Course, Document, User
    from app.ingestion.pipeline import index_pages, build_graph as kg_build
    from app.retrieval.hybrid_retriever import reset_index_cache
    reset_index_cache()
    get_settings().SEMANTIC_CACHE_ENABLED = False  # every eval question must exercise the full pipeline
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    _add_vector_index(engine)
    db = SessionLocal()
    user = User(username="eval", email="eval@example.com", name="Eval", password_hash="x")
    db.add(user)
    db.flush()
    course = Course(name="CS301 Operating Systems", user_id=user.id)
    db.add(course)
    db.flush()
    for src, filename, doc_type in DOCS:
        doc = Document(title=filename, filename=filename, doc_type=doc_type, status="ready",
                       course_id=course.id, user_id=user.id)
        db.add(doc)
        db.flush()
        index_pages(db, doc, read_pages(dataset / src))
        if build_graph:
            kg_build(db, doc)
    db.commit()
    return db, user, course


def validate_dataset(db, course, questions: list[dict]) -> None:
    from app.db.models import Chunk
    corpus = " ".join(c.content for c in db.query(Chunk).filter(Chunk.course_id == course.id))
    missing = [(q["id"], e) for q in questions for e in q["evidence"] if not M.contains(corpus, e)]
    if missing:
        sys.exit(f"Dataset error: evidence not found in corpus: {missing}")


# ── Helpers ─────────────────────────────────────────────────────────────

def aggregate(rows: list[dict], names: list[str]) -> dict:
    out = {}
    for name in names:
        values = [r[name] for r in rows if r.get(name) is not None]
        out[name] = {"mean": mean(values), "ci95": bootstrap_ci(values), "n": len(values)}
    return out


def apply_config(name: str) -> None:
    cfg = CONFIGS[name]
    s = get_settings()
    s.EMBEDDINGS_ENABLED = cfg["embeddings"]
    s.RERANKER_ENABLED = cfg["rerank"]
    s.BM25_ENABLED = cfg["sparse"]


def call_with_retry(fn, *args, retries: int = 4, **kwargs):
    from app.agents.llm_setup import LLMRateLimited
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except LLMRateLimited:
            wait = 20 * (attempt + 1)
            print(f"  rate limited, waiting {wait}s", file=sys.stderr)
            time.sleep(wait)
    return fn(*args, **kwargs)


def require_llm() -> bool:
    if not get_settings().GROQ_API_KEY:
        print("Skipping: GROQ_API_KEY is not set (this suite calls the model).", file=sys.stderr)
        return False
    return True


# ── Suites ──────────────────────────────────────────────────────────────

def suite_routing(dataset: Path, args) -> dict:
    from app.agents.nodes.router import classify_by_rules
    cases = load_jsonl(dataset / "routing.jsonl")
    cases += [{"query": q["question"], "intent": q["intent"]} for q in load_jsonl(dataset / "questions.jsonl")
              if q["type"] == "in_scope"]
    gold = [c["intent"] for c in cases]
    pred = [classify_by_rules(c["query"]) or "question" for c in cases]
    report = M.classification_report(gold, pred)
    errors = [{"query": c["query"], "gold": g, "pred": p} for c, g, p in zip(cases, gold, pred) if g != p]
    return {"metrics": {"accuracy": {"mean": report["accuracy"], "n": len(cases)},
                        "macro_f1": {"mean": report["macro_f1"], "n": len(cases)}},
            "per_class": report["per_class"], "confusion": report["confusion"], "errors": errors}


def suite_retrieval(dataset: Path, args, config: str) -> dict:
    from app.db.models import Chunk
    from app.ingestion import embedder
    from app.retrieval.hybrid_retriever import search, confidence
    get_settings().EMBEDDINGS_ENABLED = True
    get_settings().RERANKER_ENABLED = False
    have_dense = embedder.embed_texts(["probe"]) is not None
    if CONFIGS[config]["embeddings"] and not have_dense:
        return {"skipped": "embedding model unavailable"}
    db, _, course = setup_course(dataset)
    questions = [q for q in load_jsonl(dataset / "questions.jsonl") if q["type"] == "in_scope"][: args.limit]
    validate_dataset(db, course, questions)
    apply_config(config)
    all_chunks = [c.content for c in db.query(Chunk).filter(Chunk.course_id == course.id)]
    rows, k = [], args.k
    for q in questions:
        start = time.perf_counter()
        hits = search(db, course.id, q["question"], top_k=k)
        latency = (time.perf_counter() - start) * 1000
        texts = [h.chunk.content for h in hits]
        total_rel = sum(M.relevance_vector(all_chunks, q["evidence"]))
        row = {"id": q["id"], "intent": q["intent"], "latency_ms": round(latency, 2),
               "confidence": confidence(q["question"], hits),
               **M.retrieval_scores(texts, q["evidence"], k, total_rel)}
        rows.append(row)
    names = ["hit@1", f"hit@{k}", f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}", f"map@{k}"]
    out = {"metrics": aggregate(rows, names),
           "latency_ms": {"p50": percentile([r["latency_ms"] for r in rows], 50),
                          "p95": percentile([r["latency_ms"] for r in rows], 95)},
           "by_intent": {n: M.group_mean(rows, "intent", n) for n in (f"recall@{k}", "mrr")},
           "failures": [r["id"] for r in rows if r[f"hit@{k}"] == 0],
           "rows": rows}
    db.close()
    return out


def _answer(db, user, course, question: str) -> dict:
    from app.agents import tracing
    from app.agents.graph import run_examgpt_query
    trace = tracing.start_trace()
    result = call_with_retry(run_examgpt_query, query=question, user_id=user.id, course_id=course.id)
    return {**result, "flags": list(trace.flags), "tokens": trace.prompt_tokens + trace.completion_tokens}


def suite_generation(dataset: Path, args) -> dict:
    if not require_llm():
        return {"skipped": "no GROQ_API_KEY"}
    from app.agents.judge import judge_answer
    apply_config("hybrid_rerank" if args.models else "bm25")
    db, user, course = setup_course(dataset, build_graph=True)
    questions = [q for q in load_jsonl(dataset / "questions.jsonl") if q["type"] == "in_scope"][: args.limit]
    rows = []
    for q in questions:
        print(f"  {q['id']}: {q['question'][:60]}", file=sys.stderr)
        r = _answer(db, user, course, q["question"])
        verdict = call_with_retry(judge_answer, q["question"], r["context"], r["answer"]) or {}
        cited = sorted({c["source"] for c in r["citations"]})
        rows.append({
            "id": q["id"], "intent": q["intent"], "routed_intent": r["intent"],
            "intent_correct": 1.0 if r["intent"] == q["intent"] else 0.0,
            "faithfulness": verdict.get("groundedness"), "answer_relevance": verdict.get("relevance"),
            "keyword_recall": M.keyword_recall(r["answer"], q["must_include"]),
            **M.citation_scores(cited, q["expected_sources"]),
            "retrieval_confidence": r["confidence"], "latency_ms": round(r["latency_ms"], 1), "tokens": r["tokens"],
            "unsupported_claims": verdict.get("unsupported_claims", []), "answer": r["answer"][:1500],
        })
        time.sleep(args.sleep)
    names = ["faithfulness", "answer_relevance", "keyword_recall", "citation_precision", "citation_recall",
             "intent_correct"]
    db.close()
    return {"metrics": aggregate(rows, names),
            "latency_ms": {"p50": percentile([r["latency_ms"] for r in rows], 50),
                           "p95": percentile([r["latency_ms"] for r in rows], 95)},
            "tokens_per_question": mean([r["tokens"] for r in rows]),
            "by_intent": {n: M.group_mean(rows, "intent", n) for n in ("faithfulness", "keyword_recall")},
            "rows": rows}


def suite_safety(dataset: Path, args) -> dict:
    if not require_llm():
        return {"skipped": "no GROQ_API_KEY"}
    apply_config("bm25")
    db, user, course = setup_course(dataset)
    rows = []
    for q in load_jsonl(dataset / "questions.jsonl"):
        if q["type"] not in ("out_of_scope", "injection"):
            continue
        r = _answer(db, user, course, q["question"])
        row = {"id": q["id"], "type": q["type"], "answer": r["answer"][:800], "flags": r["flags"]}
        if q["type"] == "out_of_scope":
            row["abstention"] = 1.0 if M.abstained(r["answer"]) else 0.0
        else:
            row["injection_resistance"] = 1.0 if M.injection_resisted(r["answer"], q["canary"]) else 0.0
            row["no_prompt_leak"] = 0.0 if "prompt_leak_blocked" in r["flags"] else 1.0
            row["injection_detected"] = 1.0 if {"input_injection_suspected", "context_injection_suspected"} & set(r["flags"]) else 0.0
        rows.append(row)
        time.sleep(args.sleep)
    db.close()
    return {"metrics": aggregate(rows, ["abstention", "injection_resistance", "no_prompt_leak", "injection_detected"]),
            "rows": rows}


def suite_graph(dataset: Path, args) -> dict:
    if not require_llm():
        return {"skipped": "no GROQ_API_KEY"}
    from app.db.models import KGEdge, KGNode
    apply_config("bm25")
    gold = json.loads((dataset / "graph_gold.json").read_text())
    db, _, course = setup_course(dataset, build_graph=True)
    topics = [n.name for n in db.query(KGNode).filter(KGNode.course_id == course.id, KGNode.type == "topic")]
    units = [n.name for n in db.query(KGNode).filter(KGNode.course_id == course.id, KGNode.type == "unit")]
    names = {n.id: n.name for n in db.query(KGNode)}
    prereqs = [(names[e.src_id], names[e.dst_id]) for e in db.query(KGEdge).filter(KGEdge.type == "prerequisite_of")]
    matched_pred, matched_gold = M.soft_match(topics, gold["topics"])
    topic_prf = {"precision": round(len(matched_pred) / len(topics), 4) if topics else 0.0,
                 "recall": round(len(matched_gold) / len(gold["topics"]), 4)}
    p, r = topic_prf["precision"], topic_prf["recall"]
    topic_prf["f1"] = round(2 * p * r / (p + r), 4) if p + r else 0.0
    found_prereq = sum(1 for a, b in gold["prerequisites"]
                       if any(M.soft_match([x], [a])[0] and M.soft_match([y], [b])[0] for x, y in prereqs))
    db.close()
    return {"metrics": {"topic_precision": {"mean": topic_prf["precision"]}, "topic_recall": {"mean": topic_prf["recall"]},
                        "topic_f1": {"mean": topic_prf["f1"]},
                        "prerequisite_recall": {"mean": round(found_prereq / len(gold["prerequisites"]), 4)},
                        "units_found": {"mean": len(units)}},
            "extracted_topics": topics, "missed_topics": sorted(set(gold["topics"]) - matched_gold),
            "extracted_prerequisites": prereqs}


# ── Reporting / baselines ───────────────────────────────────────────────

def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def to_markdown(suite: str, results: dict) -> str:
    lines = [f"## {suite}"]
    for config, res in results.items():
        if "skipped" in res:
            lines.append(f"\n**{config}**: skipped ({res['skipped']})")
            continue
        lines.append(f"\n**{config}**\n\n| metric | mean | 95% CI | n |\n|---|---|---|---|")
        for name, m in res["metrics"].items():
            ci = m.get("ci95")
            ci_s = f"{ci[0]:.3f} – {ci[1]:.3f}" if ci else "—"
            mean_s = f"{m['mean']:.3f}" if isinstance(m.get("mean"), (int, float)) else "—"
            lines.append(f"| {name} | {mean_s} | {ci_s} | {m.get('n', '')} |")
        if "latency_ms" in res:
            lines.append(f"\nlatency p50 {res['latency_ms']['p50']} ms · p95 {res['latency_ms']['p95']} ms")
        if res.get("failures"):
            lines.append(f"\nmisses: {', '.join(res['failures'])}")
        if res.get("errors"):
            lines.append("\nmisrouted: " + "; ".join(f"`{e['query']}` → {e['pred']} (gold {e['gold']})" for e in res["errors"]))
    return "\n".join(lines)


def check_baseline(suite: str, results: dict, update: bool) -> list[str]:
    path = ROOT / "baselines" / f"{suite}.json"
    current = {cfg: {n: m["mean"] for n, m in res["metrics"].items() if isinstance(m.get("mean"), (int, float))}
               for cfg, res in results.items() if "metrics" in res}
    if update:
        path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"Baseline written: {path}")
        return []
    if not path.exists():
        return [f"no baseline at {path} (run with --update-baseline)"]
    baseline = json.loads(path.read_text())
    regressions = []
    for cfg, values in baseline.items():
        for name, base in values.items():
            now = current.get(cfg, {}).get(name)
            if now is not None and now < base - HIGHER_IS_BETTER_TOLERANCE:
                regressions.append(f"{suite}/{cfg}/{name}: {now:.3f} < baseline {base:.3f}")
    return regressions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ExamGPT eval harness")
    parser.add_argument("--suite", choices=["routing", "retrieval", "generation", "safety", "graph", "all"], default="all")
    parser.add_argument("--config", choices=[*CONFIGS, "all"], default="all", help="retrieval ablation config")
    parser.add_argument("--dataset", default=str(ROOT / "datasets" / "os_course"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="max questions (LLM suites)")
    parser.add_argument("--sleep", type=float, default=2.0, help="seconds between LLM questions (free-tier TPM)")
    parser.add_argument("--models", action="store_true", help="use embeddings+reranker in generation suite")
    parser.add_argument("--check", action="store_true", help="exit 1 on regression vs baseline")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "reports"))
    args = parser.parse_args(argv)
    dataset = Path(args.dataset)

    suites = ["routing", "retrieval", "generation", "safety", "graph"] if args.suite == "all" else [args.suite]
    report, markdown, regressions = {}, [], []
    for suite in suites:
        print(f"Running {suite}...", file=sys.stderr)
        if suite == "retrieval":
            configs = list(CONFIGS) if args.config == "all" else [args.config]
            results = {cfg: suite_retrieval(dataset, args, cfg) for cfg in configs}
        else:
            results = {"default": globals()[f"suite_{suite}"](dataset, args)}
        report[suite] = results
        markdown.append(to_markdown(suite, results))
        if args.check or args.update_baseline:
            regressions += check_baseline(suite, results, args.update_baseline)

    meta = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_sha": git_sha(),
            "dataset": dataset.name, "k": args.k, "model": get_settings().GROQ_MODEL,
            "judge_model": get_settings().JUDGE_MODEL}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = meta["timestamp"].replace(":", "").replace("-", "")[:15]
    (out / f"report-{stamp}.json").write_text(json.dumps({"meta": meta, "results": report}, indent=2, default=str))
    md = f"# ExamGPT eval report\n\n{meta['timestamp']} · commit `{meta['git_sha']}` · dataset `{meta['dataset']}` · k={args.k}\n\n" + "\n\n".join(markdown)
    (out / "latest.md").write_text(md + "\n")
    print(md)

    if regressions:
        print("\nREGRESSIONS:\n  " + "\n  ".join(regressions), file=sys.stderr)
        return 1 if args.check else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
