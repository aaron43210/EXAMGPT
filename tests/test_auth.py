import jwt
import pytest
from datetime import datetime, timedelta, timezone
from cryptography.hazmat.primitives.asymmetric import rsa
import app.api.auth as auth_api
from tests.conftest import register_and_login

SECRET = "test-secret-key-that-is-long-enough-1234567890"
CLIENT_ID = "test-client.apps.googleusercontent.com"
GOOGLE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(autouse=True)
def google_keys(monkeypatch):
    """Serve our test key instead of fetching Google's JWKS."""
    class FakeJWKS:
        def get_signing_key_from_jwt(self, token):
            return type("Key", (), {"key": GOOGLE_KEY.public_key()})()
    monkeypatch.setattr(auth_api, "_google_keys", FakeJWKS())


def google_token(email="bob@gmail.com", key=GOOGLE_KEY, **overrides):
    now = datetime.now(timezone.utc)
    claims = {"iss": "https://accounts.google.com", "aud": CLIENT_ID, "sub": "1234567890",
              "email": email, "email_verified": True, "name": "Bob Smith",
              "iat": now, "exp": now + timedelta(hours=1), **overrides}
    return jwt.encode(claims, key, algorithm="RS256")


def google_login(client, credential):
    return client.post("/api/v1/auth/google", json={"credential": credential})


def test_google_login_creates_user_then_reuses_it(client):
    r = google_login(client, google_token("Bob.Smith@Gmail.com"))
    assert r.status_code == 200, r.text
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    me = client.get("/api/v1/auth/me", headers=headers).json()
    assert me["email"] == "bob.smith@gmail.com" and me["username"] == "bob.smith" and me["name"] == "Bob Smith"
    assert "password_hash" not in me

    again = google_login(client, google_token("bob.smith@gmail.com"))
    token = again.json()["access_token"]
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).json()["id"] == me["id"]


def test_google_login_links_existing_account_by_email(client):
    old = client.get("/api/v1/auth/me", headers=register_and_login(client, "carol")).json()
    r = google_login(client, google_token("carol@example.com"))
    token = r.json()["access_token"]
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).json()["id"] == old["id"]


def test_google_username_collision_gets_suffix(client):
    register_and_login(client, "dave")  # owns username "dave" with dave@example.com
    token = google_login(client, google_token("dave@gmail.com")).json()["access_token"]
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).json()["username"] == "dave2"


def test_google_login_rejects_bad_tokens(client):
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(timezone.utc)
    for bad in [
        google_token(key=other_key),                        # forged signature
        google_token(aud="someone-else.apps.googleusercontent.com"),
        google_token(iss="https://evil.example.com"),
        google_token(exp=now - timedelta(seconds=5)),
        google_token(email_verified=False),
        jwt.encode({"email": "x@gmail.com"}, SECRET, algorithm="HS256"),  # HS256 not accepted
    ]:
        assert google_login(client, bad).status_code == 401
    assert google_login(client, "").status_code == 422


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
    # A fresh sign-in works again
    assert google_login(client, google_token("hank@example.com")).status_code == 200


def test_login_rate_limited(client):
    codes = [google_login(client, google_token(email="ivan@gmail.com", aud="wrong")).status_code
             for _ in range(12)]
    assert 429 in codes
