from tests.conftest import register_and_login, upload, make_pdf


def test_course_crud(client, auth):
    r = client.post("/api/v1/courses/", json={"name": "  DBMS  ", "description": "Databases"}, headers=auth)
    assert r.status_code == 201
    cid = r.json()["id"]
    assert r.json()["name"] == "DBMS"

    r = client.get("/api/v1/courses/", headers=auth)
    assert r.json()["total"] == 1

    assert client.get(f"/api/v1/courses/{cid}", headers=auth).status_code == 200
    assert client.delete(f"/api/v1/courses/{cid}", headers=auth).status_code == 204
    assert client.get(f"/api/v1/courses/{cid}", headers=auth).status_code == 404


def test_course_validation(client, auth):
    assert client.post("/api/v1/courses/", json={"name": ""}, headers=auth).status_code == 422
    assert client.post("/api/v1/courses/", json={"name": "x" * 500}, headers=auth).status_code == 422


def test_users_cannot_access_each_others_courses(client, auth, course, fake_llm):
    other = register_and_login(client, "mallory")
    cid = course["id"]
    assert client.get(f"/api/v1/courses/{cid}", headers=other).status_code == 404
    assert client.delete(f"/api/v1/courses/{cid}", headers=other).status_code == 404
    assert client.get(f"/api/v1/courses/{cid}/documents", headers=other).status_code == 404
    assert upload(client, other, cid, "sneaky content here").status_code == 404
    assert client.post(f"/api/v1/courses/{cid}/query", json={"query": "hi"}, headers=other).status_code == 404
    assert client.get(f"/api/v1/courses/{cid}/chat-history", headers=other).status_code == 404
    assert client.post(f"/api/v1/courses/{cid}/study-plan", json={"days": 3}, headers=other).status_code == 404
    assert client.get("/api/v1/courses/", headers=other).json()["total"] == 0


def test_upload_is_indexed_and_persisted(client, auth, course):
    r = upload(client, auth, course["id"], "Deadlock occurs when processes wait on each other.\nBanker's algorithm avoids deadlock.")
    assert r.status_code == 202, r.text
    assert r.json()["status"] == "processing"

    docs = client.get(f"/api/v1/courses/{course['id']}/documents", headers=auth).json()["documents"]
    assert docs[0]["status"] == "ready"
    assert docs[0]["chunk_count"] >= 1


def test_upload_rejects_bad_files(client, auth, course):
    cid = course["id"]
    url = f"/api/v1/courses/{cid}/documents"
    # wrong extension
    r = client.post(url, files={"file": ("notes.txt", b"hello", "text/plain")}, headers=auth)
    assert r.status_code == 400
    # .pdf extension but not a PDF
    r = client.post(url, files={"file": ("fake.pdf", b"MZ\x90\x00 not a pdf", "application/pdf")}, headers=auth)
    assert r.status_code == 400
    # empty
    r = client.post(url, files={"file": ("empty.pdf", b"", "application/pdf")}, headers=auth)
    assert r.status_code == 400
    # bad doc_type
    r = client.post(url, files={"file": ("a.pdf", make_pdf("some text here ok"), "application/pdf")},
                    data={"doc_type": "malware"}, headers=auth)
    assert r.status_code == 400


def test_upload_too_large(client, auth, course, monkeypatch):
    from app.core.config import get_settings
    monkeypatch.setattr(get_settings(), "MAX_UPLOAD_SIZE_MB", 1)
    big = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 10)
    r = client.post(f"/api/v1/courses/{course['id']}/documents",
                    files={"file": ("big.pdf", big, "application/pdf")}, headers=auth)
    assert r.status_code == 413


def test_corrupt_pdf_marked_failed_with_message(client, auth, course):
    r = client.post(f"/api/v1/courses/{course['id']}/documents",
                    files={"file": ("broken.pdf", b"%PDF-1.4\ngarbage garbage", "application/pdf")}, headers=auth)
    assert r.status_code == 202
    doc = client.get(f"/api/v1/courses/{course['id']}/documents", headers=auth).json()["documents"][0]
    assert doc["status"] == "failed"
    assert doc["error_message"]


def test_path_traversal_filename_sanitized(client, auth, course):
    r = upload(client, auth, course["id"], "Paging and segmentation basics", filename="../../etc/passwd.pdf")
    assert r.status_code == 202
    assert r.json()["filename"] == "passwd.pdf"


def test_same_filename_can_be_uploaded_twice(client, auth, course):
    assert upload(client, auth, course["id"], "first version of notes").status_code == 202
    assert upload(client, auth, course["id"], "second version of notes").status_code == 202


def test_delete_document_removes_chunks(client, auth, course):
    from app.db.database import SessionLocal
    from app.db.models import Chunk
    doc = upload(client, auth, course["id"], "Semaphores and mutexes").json()
    assert client.delete(f"/api/v1/courses/{course['id']}/documents/{doc['id']}", headers=auth).status_code == 204
    with SessionLocal() as db:
        assert db.query(Chunk).count() == 0


def test_delete_course_with_all_children(client, auth, course, fake_llm):
    cid = course["id"]
    upload(client, auth, cid, "Process scheduling: round robin, FCFS, SJF")
    client.post(f"/api/v1/courses/{cid}/query", json={"query": "What is round robin?"}, headers=auth)
    fake_llm.reply = '{"schedule": [{"day": 1, "date_label": "Day 1", "topics": []}], "tips": []}'
    assert client.post(f"/api/v1/courses/{cid}/study-plan", json={"days": 2}, headers=auth).status_code == 200
    assert client.delete(f"/api/v1/courses/{cid}", headers=auth).status_code == 204
    assert client.get("/api/v1/courses/", headers=auth).json()["total"] == 0
