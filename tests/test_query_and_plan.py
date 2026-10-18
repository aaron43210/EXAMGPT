import json
from app.agents.llm_setup import LLMRateLimited, LLMError, parse_json
from tests.conftest import register_and_login, upload


def _context_sent(fake_llm) -> str:
    return fake_llm.calls[-1]["messages"][-1]["content"]


def test_query_uses_only_own_course_documents(client, auth, course, fake_llm):
    # Another user's course mentions the same topic
    other = register_and_login(client, "eve")
    other_course = client.post("/api/v1/courses/", json={"name": "Secret"}, headers=other).json()
    upload(client, other, other_course["id"], "Deadlock SECRET-EVE-DATA private notes", doc_type="syllabus")

    upload(client, auth, course["id"], "Deadlock requires mutual exclusion, hold and wait, no preemption, circular wait.")

    r = client.post(f"/api/v1/courses/{course['id']}/query", json={"query": "What causes deadlock?"}, headers=auth)
    assert r.status_code == 200, r.text
    assert r.json()["response"] == "Mocked answer"
    sent = _context_sent(fake_llm)
    assert "circular wait" in sent
    assert "SECRET-EVE-DATA" not in sent
    assert r.json()["citations"] == [{"source": "notes.pdf", "type": "notes"}]


def test_retrieval_survives_restart(client, auth, course, fake_llm):
    """Chunks live in the DB, not process memory."""
    upload(client, auth, course["id"], "Thrashing happens when paging activity is excessive.")
    import app.retrieval.hybrid_retriever as hr
    import importlib
    importlib.reload(hr)  # simulate fresh process state
    r = client.post(f"/api/v1/courses/{course['id']}/query", json={"query": "What is thrashing?"}, headers=auth)
    assert r.status_code == 200
    assert "excessive" in _context_sent(fake_llm)


def test_query_with_no_documents_still_answers(client, auth, course, fake_llm):
    r = client.post(f"/api/v1/courses/{course['id']}/query", json={"query": "hello"}, headers=auth)
    assert r.status_code == 200
    assert r.json()["citations"] == []


def test_chat_history_and_follow_up(client, auth, course, fake_llm):
    cid = course["id"]
    client.post(f"/api/v1/courses/{cid}/query", json={"query": "Explain virtual memory"}, headers=auth)
    client.post(f"/api/v1/courses/{cid}/query", json={"query": "explain this more"}, headers=auth)
    messages = fake_llm.calls[-1]["messages"]
    assert any(m["role"] == "user" and m["content"] == "Explain virtual memory" for m in messages)

    hist = client.get(f"/api/v1/courses/{cid}/chat-history", headers=auth).json()
    assert hist["total"] == 2
    assert hist["messages"][0]["query"] == "explain this more"  # newest first

    assert client.delete(f"/api/v1/courses/{cid}/chat-history", headers=auth).status_code == 204
    assert client.get(f"/api/v1/courses/{cid}/chat-history", headers=auth).json()["total"] == 0


def test_query_validation(client, auth, course, fake_llm):
    url = f"/api/v1/courses/{course['id']}/query"
    assert client.post(url, json={"query": "   "}, headers=auth).status_code == 422
    assert client.post(url, json={"query": "x" * 2001}, headers=auth).status_code == 422
    assert client.post(url, json={}, headers=auth).status_code == 422


def test_llm_rate_limit_returns_friendly_429(client, auth, course, fake_llm):
    fake_llm.error = LLMRateLimited("429 from groq: org_abc123 secret details")
    r = client.post(f"/api/v1/courses/{course['id']}/query", json={"query": "anything"}, headers=auth)
    assert r.status_code == 429
    assert "org_abc123" not in r.text
    assert "try again" in r.json()["detail"]


def test_llm_outage_returns_503_without_leaking(client, auth, course, fake_llm):
    fake_llm.error = LLMError("connection refused to internal host 10.0.0.5")
    r = client.post(f"/api/v1/courses/{course['id']}/query", json={"query": "anything"}, headers=auth)
    assert r.status_code == 503
    assert "10.0.0.5" not in r.text


def test_unexpected_error_is_generic_500(client, auth, course, monkeypatch):
    import app.api.query as q

    def boom(**kwargs):
        raise RuntimeError("password=hunter2 stack details")
    monkeypatch.setattr(q, "run_examgpt_query", boom)
    r = client.post(f"/api/v1/courses/{course['id']}/query", json={"query": "anything"}, headers=auth)
    assert r.status_code == 500
    assert "hunter2" not in r.text
    assert "Reference" in r.json()["detail"]


def test_query_rate_limit(client, auth, course, fake_llm):
    url = f"/api/v1/courses/{course['id']}/query"
    codes = [client.post(url, json={"query": f"q{i}"}, headers=auth).status_code for i in range(12)]
    assert codes.count(200) == 10 and codes[-1] == 429


def test_evaluation_retry_loop(client, auth, course, fake_llm, monkeypatch):
    from app.core.config import get_settings
    monkeypatch.setattr(get_settings(), "EVAL_ENABLED", True)
    replies = iter([
        "draft one",
        '{"passed": false, "score": 3, "feedback": "missing detail"}',
        "draft two",
        '{"passed": true, "score": 8, "feedback": "ok"}',
    ])
    fake_llm.reply = lambda messages: next(replies)
    r = client.post(f"/api/v1/courses/{course['id']}/query", json={"query": "Explain paging"}, headers=auth)
    body = r.json()
    assert body["response"] == "draft two"
    assert body["retry_count"] == 1
    assert body["evaluation"]["score"] == 8


def test_study_plan_generation(client, auth, course, fake_llm):
    cid = course["id"]
    assert client.post(f"/api/v1/courses/{cid}/study-plan", json={"days": 5}, headers=auth).status_code == 400

    upload(client, auth, cid, "Unit 1: Processes. Unit 2: Memory.", doc_type="syllabus", filename="syl.pdf")
    plan = {"title": "Plan", "schedule": [{"day": 1, "date_label": "Day 1", "theme": "Processes",
                                          "topics": [{"name": "Processes", "duration_hours": 2, "priority": "high"}]}],
            "priority_topics": ["Processes"], "tips": ["Practice"]}
    fake_llm.reply = "```json\n" + json.dumps(plan) + "\n```"
    r = client.post(f"/api/v1/courses/{cid}/study-plan", json={"days": 5, "hours_per_day": 2}, headers=auth)
    assert r.status_code == 200, r.text
    assert r.json()["plan_data"]["total_days"] == 5
    assert "Unit 1: Processes" in fake_llm.calls[-1]["messages"][-1]["content"]

    got = client.get(f"/api/v1/courses/{cid}/study-plan", headers=auth)
    assert got.status_code == 200 and got.json()["plan_data"]["schedule"][0]["theme"] == "Processes"


def test_study_plan_long_uses_phases_and_rejects_bad_json(client, auth, course, fake_llm):
    cid = course["id"]
    upload(client, auth, cid, "Unit 1: Processes. Unit 2: Memory.")
    fake_llm.reply = "not json at all"
    r = client.post(f"/api/v1/courses/{cid}/study-plan", json={"days": 60}, headers=auth)
    assert r.status_code == 502
    assert len(fake_llm.calls) == 2  # retried once

    fake_llm.reply = json.dumps({"phases": [{"phase": 1, "name": "Basics", "days": 60}]})
    r = client.post(f"/api/v1/courses/{cid}/study-plan", json={"days": 60}, headers=auth)
    assert r.status_code == 200
    assert "phases" in r.json()["plan_data"]


def test_study_plan_validation(client, auth, course):
    url = f"/api/v1/courses/{course['id']}/study-plan"
    for body in ({"days": 0}, {"days": 365}, {"hours_per_day": 0}, {"hours_per_day": 30}):
        assert client.post(url, json=body, headers=auth).status_code == 422


def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Here you go:\n```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json('prefix {"a": 3} suffix') == {"a": 3}
    assert parse_json("nope") is None
    assert parse_json("[1, 2]") is None
    assert parse_json("") is None
