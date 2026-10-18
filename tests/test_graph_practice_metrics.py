import json
import pytest
from app.agents import tracing
from app.agents.llm_setup import LLMRateLimited, budget, chat
from app.core.config import get_settings
from app.core.stats import bootstrap_ci, mean, percentile
from app.knowledge.graph import node_key
from tests.conftest import register_and_login, upload

SYLLABUS = json.dumps({"units": [
    {"name": "Unit 1: Processes", "topics": [{"name": "Process Scheduling"}, {"name": "Deadlocks", "prerequisites": ["Process Scheduling"]}]},
    {"name": "Unit 2: Memory", "topics": [{"name": "Paging"}]},
]})
PYQS = json.dumps({"questions": [
    {"question": "Explain the Banker's algorithm.", "year": 2023, "marks": 10, "topics": ["Deadlocks"]},
    {"question": "What are the conditions for deadlock?", "year": 2022, "marks": 5, "topics": ["deadlock"]},
    {"question": "Compare FCFS and SJF.", "year": 2022, "marks": 5, "topics": ["Process Scheduling"]},
    {"question": "Explain thrashing.", "year": "n/a", "marks": None, "topics": ["Thrashing"]},
]})


def _build_graph(client, auth, course, fake_llm):
    fake_llm.replies.update({"kg_syllabus": SYLLABUS, "kg_pyq": PYQS})
    upload(client, auth, course["id"], "Unit 1 processes scheduling deadlocks. Unit 2 memory paging.",
           doc_type="syllabus", filename="syllabus.pdf")
    upload(client, auth, course["id"], "2023: Banker's algorithm. 2022: deadlock conditions; FCFS vs SJF.",
           doc_type="pyq", filename="pyq.pdf")


def test_topic_map_and_most_asked(client, auth, course, fake_llm):
    _build_graph(client, auth, course, fake_llm)
    docs = client.get(f"/api/v1/courses/{course['id']}/documents", headers=auth).json()["documents"]
    assert {d["kg_status"] for d in docs} == {"ready"}

    tm = client.get(f"/api/v1/courses/{course['id']}/topics", headers=auth).json()
    units = {u["name"]: u for u in tm["units"]}
    assert set(units) == {"Unit 1: Processes", "Unit 2: Memory"}
    deadlocks = next(t for t in units["Unit 1: Processes"]["topics"] if t["name"] == "Deadlocks")
    assert deadlocks["pyq_count"] == 2  # "deadlock" was linked to the existing "Deadlocks" topic
    assert deadlocks["years"] == [2023, 2022]
    assert deadlocks["prerequisites"] == ["Process Scheduling"]
    assert tm["most_asked"][0]["topic"] == "Deadlocks"
    assert [t["name"] for t in tm["unassigned"]] == ["Thrashing"]  # new topic created from a PYQ


def test_graph_failure_does_not_fail_upload(client, auth, course, fake_llm):
    fake_llm.error = {"kg_syllabus": LLMRateLimited("busy")}
    upload(client, auth, course["id"], "Unit 1: Processes and scheduling", doc_type="syllabus")
    doc = client.get(f"/api/v1/courses/{course['id']}/documents", headers=auth).json()["documents"][0]
    assert doc["status"] == "ready" and doc["kg_status"] == "failed"

    fake_llm.error = None
    budget.reset()  # a rate limit blocks the model for the rest of the minute; simulate it passing
    fake_llm.replies["kg_syllabus"] = SYLLABUS
    r = client.post(f"/api/v1/courses/{course['id']}/documents/{doc['id']}/rebuild-graph", headers=auth)
    assert r.status_code == 202
    doc = client.get(f"/api/v1/courses/{course['id']}/documents", headers=auth).json()["documents"][0]
    assert doc["kg_status"] == "ready"


def test_deleting_documents_and_courses_clears_graph(client, auth, course, fake_llm):
    from app.db.database import SessionLocal
    from app.db.models import KGNode, PYQQuestion
    _build_graph(client, auth, course, fake_llm)
    docs = client.get(f"/api/v1/courses/{course['id']}/documents", headers=auth).json()["documents"]
    pyq_doc = next(d for d in docs if d["doc_type"] == "pyq")
    client.delete(f"/api/v1/courses/{course['id']}/documents/{pyq_doc['id']}", headers=auth)
    with SessionLocal() as db:
        assert db.query(PYQQuestion).count() == 0
        assert db.query(KGNode).filter(KGNode.name == "Thrashing").count() == 0
    assert client.delete(f"/api/v1/courses/{course['id']}", headers=auth).status_code == 204
    with SessionLocal() as db:
        assert db.query(KGNode).count() == 0


def test_important_questions_use_pyq_stats(client, auth, course, fake_llm):
    _build_graph(client, auth, course, fake_llm)
    r = client.post(f"/api/v1/courses/{course['id']}/query", json={"query": "most important questions"}, headers=auth)
    assert r.json()["intent"] == "important_questions"
    context = fake_llm.by_purpose("exam_analyst")[-1]["messages"][-1]["content"]
    assert "- Deadlocks: asked 2x — 2023, 2022" in context


def test_study_plan_uses_topic_map(client, auth, course, fake_llm):
    _build_graph(client, auth, course, fake_llm)
    fake_llm.replies["planner"] = json.dumps({"schedule": [{"day": 1, "topics": []}]})
    assert client.post(f"/api/v1/courses/{course['id']}/study-plan", json={"days": 3}, headers=auth).status_code == 200
    prompt = fake_llm.by_purpose("planner")[-1]["messages"][-1]["content"]
    assert "  - Deadlocks (pyq x2; after Process Scheduling)" in prompt


def test_node_key_normalization():
    assert node_key("Unit 3: Memory Management") == "memory management"
    assert node_key("  Deadlocks ") == node_key("deadlocks")
    assert node_key("Module II - Paging") == "paging"


# ── Practice: quiz + grading ────────────────────────────────────────────

def test_structured_quiz_validates_questions(client, auth, course, fake_llm):
    upload(client, auth, course["id"], "Deadlock needs circular wait and mutual exclusion.")
    fake_llm.replies["quiz_master"] = json.dumps({"questions": [
        {"question": "Which is a deadlock condition?", "options": ["Circular wait", "Paging", "Caching", "DMA"],
         "answer_index": 0, "explanation": "Coffman", "source": "notes.pdf"},
        {"question": "Bad: 3 options", "options": ["a", "b", "c"], "answer_index": 0},
        {"question": "Bad index", "options": ["a", "b", "c", "d"], "answer_index": 7},
        {"question": "Invented source", "options": ["a", "b", "c", "d"], "answer_index": 1, "source": "made-up.pdf"},
    ]})
    r = client.post(f"/api/v1/courses/{course['id']}/quiz", json={"topic": "deadlock", "num_questions": 5}, headers=auth)
    assert r.status_code == 200, r.text
    qs = r.json()["questions"]
    assert len(qs) == 2
    assert qs[0]["source"] == "notes.pdf" and qs[1]["source"] is None
    assert fake_llm.by_purpose("quiz_master")[-1]["json_mode"] is True

    fake_llm.replies["quiz_master"] = "not json"
    assert client.post(f"/api/v1/courses/{course['id']}/quiz", json={}, headers=auth).status_code == 502
    assert client.post(f"/api/v1/courses/{course['id']}/quiz", json={"num_questions": 50}, headers=auth).status_code == 422


def test_grading_clamps_and_stores_result(client, auth, course, fake_llm):
    from app.db.database import SessionLocal
    from app.db.models import EvaluationResult
    fake_llm.replies["grader"] = json.dumps({"score": 14, "max_score": 10, "strengths": ["clear"],
                                             "missing_points": ["no example"], "feedback": "Good", "model_answer": "..."})
    r = client.post(f"/api/v1/courses/{course['id']}/grade",
                    json={"question": "Define deadlock", "answer": "Processes waiting forever", "max_score": 10},
                    headers=auth)
    assert r.status_code == 200
    assert r.json()["score"] == 10.0  # clamped to max
    with SessionLocal() as db:
        row = db.query(EvaluationResult).one()
        assert (row.score, row.max_score, row.course_id) == (10, 10, course["id"])
    m = client.get(f"/api/v1/courses/{course['id']}/metrics", headers=auth).json()
    assert m["grading"] == {"answers_graded": 1, "mean_score_pct": 1.0}

    fake_llm.replies["grader"] = '{"feedback": "no score"}'
    r = client.post(f"/api/v1/courses/{course['id']}/grade", json={"question": "Define deadlock", "answer": "x"},
                    headers=auth)
    assert r.status_code == 502


# ── Metrics + online eval ───────────────────────────────────────────────

def test_metrics_aggregation(client, auth, course, fake_llm):
    for q in ("hello", "What is paging?", "quiz me"):
        client.post(f"/api/v1/courses/{course['id']}/query", json={"query": q}, headers=auth)
    m = client.get(f"/api/v1/courses/{course['id']}/metrics", headers=auth).json()
    assert m["requests"]["total"] == 3
    assert m["requests"]["by_intent"] == {"chitchat": 1, "question": 1, "quiz": 1}
    assert m["tokens"] == {"prompt_total": 300, "completion_total": 60, "per_request_mean": 120.0}
    assert m["latency_ms"]["p50"] is not None
    assert "router" in m["agent_latency_p50_ms"]

    other = register_and_login(client, "otheruser")
    assert client.get(f"/api/v1/courses/{course['id']}/metrics", headers=other).status_code == 404


def test_admin_metrics_requires_admin(client, auth):
    from app.db.database import SessionLocal
    from app.db.models import User
    assert client.get("/api/v1/admin/metrics", headers=auth).status_code == 403
    with SessionLocal() as db:
        db.query(User).update({"role": "admin"})
        db.commit()
    assert client.get("/api/v1/admin/metrics", headers=auth).status_code == 200


def test_online_eval_scores_sampled_answers(client, auth, course, fake_llm, monkeypatch):
    from app.db.database import SessionLocal
    from app.db.models import PipelineRun
    monkeypatch.setattr(get_settings(), "ONLINE_EVAL_SAMPLE_RATE", 1.0)
    fake_llm.replies["judge"] = json.dumps({"claims": [{"claim": "a", "supported": True},
                                                       {"claim": "b", "supported": True},
                                                       {"claim": "c", "supported": False}], "relevance": 5})
    client.post(f"/api/v1/courses/{course['id']}/query", json={"query": "What is paging?"}, headers=auth)
    judge_call = fake_llm.by_purpose("judge")[0]
    assert judge_call["model"] == get_settings().JUDGE_MODEL  # a different model judges the answer
    with SessionLocal() as db:
        assert db.query(PipelineRun).one().groundedness == pytest.approx(0.667)
    m = client.get(f"/api/v1/courses/{course['id']}/metrics", headers=auth).json()
    assert m["online_eval"]["sampled"] == 1


# ── LLM gateway ─────────────────────────────────────────────────────────

def test_gateway_falls_back_when_primary_rate_limited(fake_llm):
    s = get_settings()
    fake_llm.reply = lambda messages, purpose: "ok"
    calls = []

    def flaky(messages, model, json_mode, max_tokens, temperature, purpose="general"):
        calls.append(model)
        if model == s.GROQ_MODEL:
            raise LLMRateLimited("429")
        return "from fallback", {"prompt_tokens": 1, "completion_tokens": 1}

    import app.agents.llm_setup as gw
    original, gw._complete = gw._complete, flaky
    try:
        trace = tracing.start_trace()
        assert chat([{"role": "user", "content": "hi"}]) == "from fallback"
        assert calls == [s.GROQ_MODEL, s.GROQ_FALLBACK_MODEL]
        assert trace.models == [s.GROQ_FALLBACK_MODEL]
        # primary is now blocked for the window, so the next call goes straight to fallback
        calls.clear()
        chat([{"role": "user", "content": "hi"}])
        assert calls == [s.GROQ_FALLBACK_MODEL]
    finally:
        gw._complete = original


def test_gateway_fails_fast_when_budget_exhausted(fake_llm):
    s = get_settings()
    budget.record(s.GROQ_MODEL, s.GROQ_TPM_LIMIT)
    budget.record(s.GROQ_FALLBACK_MODEL, s.GROQ_FALLBACK_TPM_LIMIT)
    with pytest.raises(LLMRateLimited):
        chat([{"role": "user", "content": "hi"}])
    assert fake_llm.calls == []  # never hit the API


# ── Stats helpers ───────────────────────────────────────────────────────

def test_stats_helpers():
    import numpy as np
    data = [12.0, 5.0, 7.5, 100.0, 42.0, 3.0]
    for p in (0, 10, 50, 90, 95, 100):
        assert percentile(data, p) == pytest.approx(np.percentile(data, p), abs=1e-4)
    assert percentile([], 50) is None and mean([]) is None
    assert mean([1, 2, None, 3]) == 2.0
    lo, hi = bootstrap_ci([0.8, 0.9, 1.0, 0.7, 0.95, 0.85])
    assert lo <= 0.866 <= hi
    assert bootstrap_ci([1.0]) is None
