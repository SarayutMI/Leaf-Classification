# tests/auth/test_session.py
"""Admin login against the platform `users` table, and Bearer authentication."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.auth import security
from src.auth.router import router
from src.config import settings

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)

PASSWORD = "hunter2"
UUID = "019ea0de-e2f5-72ce-a626-3ce3560e399d"


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setattr(settings, "JWT_SECRET", "test-secret-not-a-real-key")


def _user(**overrides):
    user = {
        "id": UUID,
        "username": "admin",
        "name": "Administrator",
        "role": "admin",
        "password": security.hash_password(PASSWORD),
        "is_active": 1,
    }
    user.update(overrides)
    return user


def _login(user, password=PASSWORD, username="admin"):
    with patch("src.auth.router.auth_repository") as mock_repo:
        mock_repo.get_admin_user_by_username.return_value = user
        return client.post(
            "/api/admin/login", json={"username": username, "password": password}
        )


# ── Tokens ────────────────────────────────────────────────
def test_token_round_trips_a_uuid_subject():
    token = security.create_access_token(UUID, "admin", "admin")
    payload = security.decode_access_token(token)

    assert payload["sub"] == UUID
    assert payload["username"] == "admin"
    assert payload["role"] == "admin"


def test_expired_token_is_rejected():
    expired = jwt.encode(
        {"sub": UUID, "exp": datetime.now(timezone.utc) - timedelta(minutes=1)},
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM,
    )
    assert security.decode_access_token(expired) is None


def test_token_signed_with_another_key_is_rejected():
    forged = jwt.encode(
        {"sub": UUID, "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
        "some-other-secret",
        algorithm=settings.JWT_ALGORITHM,
    )
    assert security.decode_access_token(forged) is None


def test_garbage_token_is_rejected():
    assert security.decode_access_token("not-a-jwt") is None


def test_a_laravel_bcrypt_hash_verifies():
    """`users.password` is written by Laravel with a $2y$ prefix."""
    hashed = security.hash_password(PASSWORD).replace("$2b$", "$2y$", 1)

    assert hashed.startswith("$2y$")
    assert security.verify_password(PASSWORD, hashed)


# ── Login ─────────────────────────────────────────────────
def test_login_returns_a_bearer_token_and_no_cookie():
    response = _login(_user())

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["token_type"] == "bearer"
    assert data["user"] == {
        "id": UUID, "username": "admin", "name": "Administrator", "role": "admin",
    }
    assert security.decode_access_token(data["access_token"])["sub"] == UUID
    # Bearer-only: the cookie transport is gone.
    assert "set-cookie" not in response.headers


def test_login_reads_the_users_table_not_classify_user():
    with patch("src.auth.router.auth_repository") as mock_repo:
        mock_repo.get_admin_user_by_username.return_value = _user()
        client.post("/api/admin/login", json={"username": "admin", "password": PASSWORD})

    mock_repo.get_admin_user_by_username.assert_called_once_with("admin")
    mock_repo.get_user_by_username.assert_not_called()


@pytest.mark.parametrize("user,password", [
    (_user(), "wrong-password"),
    (None, PASSWORD),
    (_user(role="user"), PASSWORD),
    (_user(is_active=0), PASSWORD),
])
def test_every_rejection_is_the_same_401(user, password):
    """A caller must not learn which admin accounts exist or are disabled."""
    response = _login(user, password=password)

    assert response.status_code == 401
    assert response.json()["message"] == "Invalid credentials"


def test_login_never_creates_a_user():
    with patch("src.auth.router.auth_repository") as mock_repo:
        mock_repo.get_admin_user_by_username.return_value = None
        client.post("/api/admin/login", json={"username": "nobody", "password": "x"})

    mock_repo.create_user.assert_not_called()


# ── Bearer authentication ─────────────────────────────────
def test_me_requires_a_token():
    assert client.get("/api/admin/me").status_code == 401


def test_me_rejects_a_garbage_token():
    response = client.get(
        "/api/admin/me", headers={"Authorization": "Bearer not-a-jwt"}
    )
    assert response.status_code == 401


def test_me_returns_the_token_identity():
    token = security.create_access_token(UUID, "admin", "admin")
    response = client.get("/api/admin/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json()["data"] == {
        "id": UUID, "username": "admin", "role": "admin",
    }


def test_authenticated_responses_are_not_cacheable():
    token = security.create_access_token(UUID, "admin", "admin")
    response = client.get("/api/admin/me", headers={"Authorization": f"Bearer {token}"})

    assert "no-store" in response.headers["cache-control"]


def test_logout_holds_no_server_state():
    """Nothing to revoke — the endpoint exists so the client has one call."""
    response = client.post("/api/admin/logout")

    assert response.status_code == 200
    assert "set-cookie" not in response.headers
