import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from routes.auth import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)

MOCK_HASHED = "$2b$12$mockedhashvalue"
MOCK_API_KEY = "abc123def456abc123def456abc123de"


def test_register_new_user_returns_api_key():
    with patch("routes.auth.database") as mock_db, \
         patch("routes.auth.security") as mock_sec:
        mock_db.get_user_by_username.return_value = None
        mock_sec.hash_password.return_value = MOCK_HASHED
        mock_db.create_user.return_value = 1
        mock_sec.generate_api_key.return_value = MOCK_API_KEY
        mock_db.create_token.return_value = None

        response = client.post("/api/genToken", json={"username": "alice", "password": "pass123"})

        assert response.status_code == 200
        body = response.json()
        assert body["code"] == 200
        assert body["status"] == "success"
        assert body["data"]["api_key"] == MOCK_API_KEY


def test_login_existing_user_correct_password():
    with patch("routes.auth.database") as mock_db, \
         patch("routes.auth.security") as mock_sec:
        mock_db.get_user_by_username.return_value = {
            "id": 1, "username": "alice", "password_hash": MOCK_HASHED
        }
        mock_sec.verify_password.return_value = True
        mock_db.get_token_by_user_id.return_value = {"api_key": MOCK_API_KEY}

        response = client.post("/api/genToken", json={"username": "alice", "password": "pass123"})

        assert response.status_code == 200
        assert response.json()["data"]["api_key"] == MOCK_API_KEY


def test_login_existing_user_wrong_password():
    with patch("routes.auth.database") as mock_db, \
         patch("routes.auth.security") as mock_sec:
        mock_db.get_user_by_username.return_value = {
            "id": 1, "username": "alice", "password_hash": MOCK_HASHED
        }
        mock_sec.verify_password.return_value = False

        response = client.post("/api/genToken", json={"username": "alice", "password": "wrongpass"})

        assert response.status_code == 401
        assert response.json()["message"] == "Invalid credentials"


def test_db_error_returns_500():
    with patch("routes.auth.database") as mock_db:
        mock_db.get_user_by_username.side_effect = Exception("connection refused")

        response = client.post("/api/genToken", json={"username": "alice", "password": "pass123"})

        assert response.status_code == 500
        assert response.json()["message"] == "Database error"
