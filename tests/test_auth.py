import jwt
from datetime import datetime, timedelta, timezone
from tests.conftest import register_and_login

SECRET = "test-secret-key-that-is-long-enough-1234567890"


def test_register_login_me(client):
    headers = register_and_login(client, "Bob")
    r = client.get("/api/v1/auth/me", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["username"] == "bob"  # normalized
    assert "password_hash" not in body


def test_login_with_email_and_case_insensitive(client):
    register_and_login(client, "carol")
    r = client.post("/api/v1/auth/login", data={"username": "CAROL@example.com", "password": "Passw0rd!"})
    assert r.status_code == 200


def test_duplicate_username_or_email_rejected(client):
    register_and_login(client, "dave")
    r = client.post("/api/v1/auth/register", json={
        "username": "DAVE", "email": "other@example.com", "name": "D", "password": "Passw0rd!"})
    assert r.status_code == 400
    r = client.post("/api/v1/auth/register", json={
        "username": "dave2", "email": "Dave@Example.com", "name": "D", "password": "Passw0rd!"})
    assert r.status_code == 400


def test_register_validation(client):
    base = {"username": "erin", "email": "erin@example.com", "name": "Erin", "password": "Passw0rd!"}
    for field, value in [
        ("email", "not-an-email"),
        ("password", "short1"),
        ("password", "nodigitsatall"),
        ("password", "a1" * 40),
        ("username", "ab"),
        ("username", "bad name!"),
        ("name", "   "),
    ]:
        r = client.post("/api/v1/auth/register", json={**base, field: value})
        assert r.status_code == 422, (field, value, r.text)
        assert isinstance(r.json()["detail"], str)


def test_wrong_password_and_unknown_user(client):
    register_and_login(client, "frank")
    assert client.post("/api/v1/auth/login", data={"username": "frank", "password": "nope12345"}).status_code == 401
    assert client.post("/api/v1/auth/login", data={"username": "ghost", "password": "nope12345"}).status_code == 401


def test_protected_routes_require_token(client):
    assert client.get("/api/v1/courses/").status_code == 401
    assert client.get("/api/v1/courses/", headers={"Authorization": "Bearer garbage"}).status_code == 401


def test_forged_and_expired_tokens_rejected(client):
    headers = register_and_login(client, "gina")
    user_id = client.get("/api/v1/auth/me", headers=headers).json()["id"]
    now = datetime.now(timezone.utc)

    forged = jwt.encode({"sub": str(user_id), "type": "access", "ver": 0, "exp": now + timedelta(hours=1)},
                        "wrong-secret", algorithm="HS256")
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401

    expired = jwt.encode({"sub": str(user_id), "type": "access", "ver": 0, "exp": now - timedelta(seconds=5)},
                         SECRET, algorithm="HS256")
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired}"}).status_code == 401

    none_alg = jwt.encode({"sub": str(user_id), "type": "access", "ver": 0, "exp": now + timedelta(hours=1)},
                          None, algorithm="none")
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {none_alg}"}).status_code == 401

    wrong_type = jwt.encode({"sub": str(user_id), "type": "refresh", "ver": 0, "exp": now + timedelta(hours=1)},
                            SECRET, algorithm="HS256")
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {wrong_type}"}).status_code == 401


def test_logout_revokes_existing_tokens(client):
    headers = register_and_login(client, "hank")
    assert client.post("/api/v1/auth/logout", headers=headers).status_code == 200
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 401
    # A fresh login works again
    register_headers = client.post("/api/v1/auth/login", data={"username": "hank", "password": "Passw0rd!"})
    assert register_headers.status_code == 200


def test_login_rate_limited(client):
    register_and_login(client, "ivan")
    codes = [client.post("/api/v1/auth/login", data={"username": "ivan", "password": "wrong1234"}).status_code
             for _ in range(12)]
    assert 429 in codes
