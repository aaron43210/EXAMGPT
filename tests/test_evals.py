import json
import math
import pytest
from evals import metrics as M


def test_retrieval_metrics_hand_computed():
    chunks = ["noise one", "the answer is FOO here", "more noise", "BAR fact", "noise"]
    evidence = ["answer is foo", "bar fact", "missing fact"]
    rels = M.relevance_vector(chunks, evidence)
    assert rels == [0, 1, 0, 1, 0]
    assert M.hit_at_k(rels, 1) == 0.0 and M.hit_at_k(rels, 2) == 1.0
    assert M.recall_at_k(chunks, evidence, 5) == pytest.approx(2 / 3)
    assert M.recall_at_k(chunks, evidence, 2) == pytest.approx(1 / 3)
    assert M.precision_at_k(rels, 5) == pytest.approx(0.4)
    assert M.reciprocal_rank(rels) == 0.5
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3)
    assert M.ndcg_at_k(rels, 5) == pytest.approx(dcg / idcg)
    assert M.average_precision_at_k(rels, 5) == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert M.ndcg_at_k([0, 0], 2) == 0.0 and M.reciprocal_rank([]) == 0.0


def test_ndcg_uses_total_relevant_for_ideal():
    # one relevant chunk retrieved at rank 1 but 2 exist in the corpus -> not perfect
    assert M.ndcg_at_k([1, 0, 0], 3, total_relevant=2) == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_classification_report():
    gold = ["quiz", "quiz", "question", "summary"]
    pred = ["quiz", "question", "question", "summary"]
    r = M.classification_report(gold, pred)
    assert r["accuracy"] == 0.75
    assert r["per_class"]["quiz"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(0.6667, abs=1e-4), "support": 2}
    assert r["per_class"]["question"]["precision"] == 0.5
    assert r["confusion"]["quiz"] == {"quiz": 1, "question": 1}
    assert r["macro_f1"] == pytest.approx((0.6667 + 0.6667 + 1.0) / 3, abs=1e-3)


def test_generation_and_safety_metrics():
    assert M.token_f1("the cat sat", "the cat sat") == 1.0
    assert M.token_f1("cat", "dog") == 0.0
    assert M.token_f1("a b c d", "a b") == pytest.approx(2 * 0.5 * 1 / 1.5)
    assert M.keyword_recall("Uses Hold and Wait", ["hold and wait", "circular wait"]) == 0.5
    assert M.keyword_recall("x", []) is None
    assert M.citation_scores(["a.pdf", "b.pdf"], ["a.pdf"]) == {"citation_precision": 0.5, "citation_recall": 1.0}
    assert M.citation_scores([], ["a.pdf"]) == {"citation_precision": 0.0, "citation_recall": 1.0 * 0}
    assert M.abstained("This is not covered in your course material.")
    assert M.abstained("Canberra is the capital (general knowledge).")
    assert not M.abstained("Deadlock requires circular wait.")
    assert M.injection_resisted("An interrupt vector is a table", "PINEAPPLE-7731")
    assert not M.injection_resisted("pineapple-7731", "PINEAPPLE-7731")


def test_soft_match_and_prf():
    pred, gold = M.soft_match(["Banker's Algorithm", "Paging", "Quantum Computing"],
                              ["bankers algorithm", "paging", "thrashing"])
    assert pred == {"Banker's Algorithm", "Paging"} and gold == {"bankers algorithm", "paging"}
    assert M.set_prf({"a", "b"}, {"b", "c"}) == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


# ── Harness end-to-end (fake model; real retrieval) ─────────────────────

@pytest.fixture
def harness(tmp_path, monkeypatch):
    from evals import run
    monkeypatch.setattr(run, "ROOT", tmp_path)
    (tmp_path / "baselines").mkdir()
    return run


def _args(harness, **kw):
    import argparse
    base = dict(k=5, limit=None, sleep=0, models=False)
    base.update(kw)
    return argparse.Namespace(**base)


DATASET = __import__("pathlib").Path(__file__).resolve().parent.parent / "evals" / "datasets" / "os_course"


def test_harness_routing_and_retrieval(harness, tmp_path):
    assert harness.main(["--suite", "routing", "--dataset", str(DATASET), "--out", str(tmp_path / "r")]) == 0
    res = harness.suite_retrieval(DATASET, _args(harness), "bm25")
    assert res["metrics"]["hit@5"]["n"] == 28
    assert res["metrics"]["mrr"]["mean"] > 0.8
    assert harness.suite_retrieval(DATASET, _args(harness), "dense") == {"skipped": "embedding model unavailable"} \
        or "metrics" in harness.suite_retrieval(DATASET, _args(harness), "dense")
    assert (tmp_path / "r" / "latest.md").read_text().startswith("# ExamGPT eval report")


def test_harness_baseline_regression_gate(harness, tmp_path):
    results = {"bm25": {"metrics": {"mrr": {"mean": 0.80}}}}
    (tmp_path / "baselines" / "retrieval.json").write_text(json.dumps({"bm25": {"mrr": 0.90}}))
    assert harness.check_baseline("retrieval", results, update=False) == ["retrieval/bm25/mrr: 0.800 < baseline 0.900"]
    results["bm25"]["metrics"]["mrr"]["mean"] = 0.89  # within tolerance
    assert harness.check_baseline("retrieval", results, update=False) == []


def test_harness_llm_suites_with_fake_model(harness, fake_llm):
    fake_llm.replies.update({
        "tutor": "Need = Max - Allocation, from [notes.pdf]. This is not covered beyond that.",
        "judge": json.dumps({"claims": [{"claim": "x", "supported": True}, {"claim": "y", "supported": False}],
                             "relevance": 5}),
        "kg_syllabus": json.dumps({"units": [{"name": "Unit 3: Deadlocks", "topics": [
            {"name": "Banker's Algorithm", "prerequisites": []}, {"name": "Paging"},
            {"name": "Virtual Memory", "prerequisites": ["Paging"]}]}]}),
        "kg_pyq": json.dumps({"questions": [{"question": "Explain Banker's", "year": 2023, "topics": ["Banker's Algorithm"]}]}),
    })
    gen = harness.suite_generation(DATASET, _args(harness, limit=3))
    assert gen["metrics"]["faithfulness"]["mean"] == 0.5
    assert gen["metrics"]["answer_relevance"]["mean"] == 1.0
    assert gen["metrics"]["intent_correct"]["mean"] == 1.0
    assert gen["tokens_per_question"] > 0

    safety = harness.suite_safety(DATASET, _args(harness))
    assert safety["metrics"]["abstention"]["mean"] == 1.0
    assert safety["metrics"]["injection_resistance"]["mean"] == 1.0
    assert safety["metrics"]["injection_detected"]["mean"] > 0  # the poisoned chunk / injected query get flagged

    graph = harness.suite_graph(DATASET, _args(harness))
    assert graph["metrics"]["topic_precision"]["mean"] == 1.0
    assert graph["metrics"]["prerequisite_recall"]["mean"] == 0.5  # found paging -> virtual memory only
