# tests/auth/test_session.py
"""Admin session tokens and the cookie login used by the rule-base page."""
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

# hash of "hunter2"
PASSWORD = "hunter2"


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setattr(settings, "JWT_SECRET", "test-secret-not-a-real-key")


@pytest.fixture
def user():
    return {
        "id": 7,
        "username": "admin",
        "password_hash": security.hash_password(PASSWORD),
    }


def test_token_round_trips():
    token = security.create_access_token(7, "admin")
    payload = security.decode_access_token(token)

    assert payload["sub"] == "7"
    assert payload["username"] == "admin"


def test_expired_token_is_rejected():
    expired = jwt.encode(
        {
            "sub": "7",
            "username": "admin",
            "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
        },
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM,
    )
    assert security.decode_access_token(expired) is None


def test_token_signed_with_another_key_is_rejected():
    forged = jwt.encode(
        {"sub": "7", "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
        "some-other-secret",
        algorithm=settings.JWT_ALGORITHM,
    )
    assert security.decode_access_token(forged) is None


def test_garbage_token_is_rejected():
    assert security.decode_access_token("not-a-jwt") is None


def test_login_sets_an_httponly_cookie(user):
    with patch("src.auth.router.auth_repository") as mock_repo:
        mock_repo.get_user_by_username.return_value = user
        response = client.post(
            "/api/admin/login",
            json={"username": "admin", "password": PASSWORD},
        )

    assert response.status_code == 200
    cookie = response.headers["set-cookie"]
    assert settings.SESSION_COOKIE_NAME in cookie
    assert "HttpOnly" in cookie


def test_wrong_password_is_401(user):
    with patch("src.auth.router.auth_repository") as mock_repo:
        mock_repo.get_user_by_username.return_value = user
        response = client.post(
            "/api/admin/login",
            json={"username": "admin", "password": "wrong"},
        )

    assert response.status_code == 401
    assert "set-cookie" not in response.headers


def test_unknown_username_is_401_and_creates_no_user():
    """Unlike /api/genToken, admin login must never auto-register."""
    with patch("src.auth.router.auth_repository") as mock_repo:
        mock_repo.get_user_by_username.return_value = None
        response = client.post(
            "/api/admin/login",
            json={"username": "nobody", "password": "whatever"},
        )

    assert response.status_code == 401
    mock_repo.create_user.assert_not_called()


def test_me_requires_a_session():
    # A fresh client, so no cookie left over from an earlier login test.
    anonymous = TestClient(app, raise_server_exceptions=False)
    assert anonymous.get("/api/admin/me").status_code == 401


def test_me_returns_the_logged_in_username(user):
    with patch("src.auth.router.auth_repository") as mock_repo:
        mock_repo.get_user_by_username.return_value = user
        client.post("/api/admin/login", json={"username": "admin", "password": PASSWORD})

    response = client.get("/api/admin/me")
    assert response.status_code == 200
    assert response.json()["data"]["username"] == "admin"


def test_logout_clears_the_cookie():
    response = client.post("/api/admin/logout")
    assert response.status_code == 200
    assert settings.SESSION_COOKIE_NAME in response.headers["set-cookie"]
