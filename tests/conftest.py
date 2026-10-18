import os

# Configure before the app is imported
os.environ.update({
    "ENVIRONMENT": "test",
    "DATABASE_URL": os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:"),
    "SECRET_KEY": "test-secret-key-that-is-long-enough-1234567890",
    "GROQ_API_KEY": "test-key",
    "EMBEDDINGS_ENABLED": os.getenv("TEST_EMBEDDINGS", "false"),
    "EVAL_ENABLED": "false",
    "FRONTEND_URL": "https://examgpt.netlify.app",
})

import pytest
from fastapi.testclient import TestClient
from app.db.database import Base, engine
from app.core.rate_limit import limiter
from app.main import app


def make_pdf(text: str) -> bytes:
    """Build a minimal single-page PDF containing `text` (one line per \\n)."""
    lines = text.split("\n")
    ops = ["BT", "/F1 11 Tf", "14 TL", "50 780 Td"]
    for line in lines:
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"({safe}) Tj T*")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.create_all(bind=engine)
    limiter.reset()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace Groq with a recorder. Set `.reply` / `.error` to control behaviour."""
    class Fake:
        reply = "Mocked answer"
        error = None
        calls: list = []

        def __call__(self, messages, **kwargs):
            self.calls.append({"messages": messages, **kwargs})
            if self.error:
                raise self.error
            return self.reply(messages) if callable(self.reply) else self.reply

    fake = Fake()
    fake.calls = []
    import app.agents.nodes.answer as answer
    import app.agents.nodes.evaluation as evaluation
    import app.api.study_plan as study_plan
    for mod in (answer, evaluation, study_plan):
        monkeypatch.setattr(mod, "chat", fake)
    return fake


def register_and_login(client, username="alice", password="Passw0rd!"):
    r = client.post("/api/v1/auth/register", json={
        "username": username, "email": f"{username}@example.com", "name": username.title(), "password": password,
    })
    assert r.status_code == 201, r.text
    r = client.post("/api/v1/auth/login", data={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def auth(client):
    return register_and_login(client)


@pytest.fixture
def course(client, auth):
    r = client.post("/api/v1/courses/", json={"name": "Operating Systems"}, headers=auth)
    assert r.status_code == 201
    return r.json()


def upload(client, headers, course_id, text, doc_type="notes", filename="notes.pdf"):
    return client.post(
        f"/api/v1/courses/{course_id}/documents",
        files={"file": (filename, make_pdf(text), "application/pdf")},
        data={"doc_type": doc_type},
        headers=headers,
    )
