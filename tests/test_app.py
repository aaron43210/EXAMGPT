import pytest
from app.core.config import Settings
from app.ingestion.chunker import Chunker
from app.api.documents import safe_filename


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["status"] == "ready"


def test_security_headers(client):
    r = client.get("/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert "max-age" in r.headers["Strict-Transport-Security"]
    assert r.headers["X-Request-ID"]


def test_cors_allows_frontend_only(client):
    ok = client.options("/api/v1/courses/", headers={
        "Origin": "https://examgpt.netlify.app", "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "https://examgpt.netlify.app"
    bad = client.options("/api/v1/courses/", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in bad.headers


def test_docs_disabled_by_default(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_production_config_guard():
    with pytest.raises(RuntimeError) as e:
        Settings(ENVIRONMENT="production", _env_file=None, SECRET_KEY="short", DATABASE_URL="sqlite:///x.db",
                 GROQ_API_KEY="", FRONTEND_URL="").validate_for_production()
    msg = str(e.value)
    assert "SECRET_KEY" in msg and "DATABASE_URL" in msg and "GROQ_API_KEY" in msg

    Settings(
        ENVIRONMENT="production", _env_file=None,
        SECRET_KEY="x" * 40, DATABASE_URL="postgresql://u:p@h/db",
        GROQ_API_KEY="k", FRONTEND_URL="https://examgpt.netlify.app",
    ).validate_for_production()


def test_production_cors_excludes_localhost():
    s = Settings(ENVIRONMENT="production", FRONTEND_URL="https://examgpt.netlify.app/", _env_file=None)
    assert s.cors_origins == ["https://examgpt.netlify.app"]


def test_postgres_url_scheme_normalized():
    assert Settings(DATABASE_URL="postgres://a:b@h:5432/d", _env_file=None).sqlalchemy_url.startswith("postgresql://")


def test_chunker_overlap_and_bounds():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = Chunker(chunk_size=500, chunk_overlap=50).chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    assert Chunker().chunk_text("") == []


@pytest.mark.parametrize("raw,expected", [
    ("notes.pdf", "notes.pdf"),
    ("../../etc/passwd.PDF", "passwd.pdf"),
    ("C:\\Users\\x\\file.pdf", "file.pdf"),
    ("<script>.pdf", "script_.pdf"),
    ("", "document.pdf"),
    (None, "document.pdf"),
])
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected


def test_migrations_apply_cleanly(tmp_path, monkeypatch):
    """alembic upgrade head works on a fresh database and matches the models."""
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect
    from app.core import config as cfg

    url = f"sqlite:///{tmp_path / 'm.db'}"
    monkeypatch.setattr(cfg.get_settings(), "DATABASE_URL", url)
    command.upgrade(Config("alembic.ini"), "head")
    cols = {c["name"] for c in inspect(create_engine(url)).get_columns("documents")}
    assert {"content_text", "error_message", "chunk_count"} <= cols
    assert "chunks" in inspect(create_engine(url)).get_table_names()
