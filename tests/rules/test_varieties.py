# tests/rules/test_varieties.py
"""Yam varieties under rule groups. The repository is mocked — no database."""
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from src.auth import security
from src.config import settings
from src.main import validation_error_handler
from src.rules.router import router

app = FastAPI()
app.include_router(router)
# The indexed 422 shape is installed on the real app; mirror it here.
app.add_exception_handler(RequestValidationError, validation_error_handler)

VARIETY = {"id": 1, "group_id": 1, "name": "มันเสือ", "is_active": True}
BATCH = {"group_id": 1, "items": [{"name": "มันเสือ", "is_active": True}]}


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setattr(settings, "JWT_SECRET", "test-secret-not-a-real-key")


@pytest.fixture
def client():
    c = TestClient(app, raise_server_exceptions=False)
    token = security.create_access_token("uuid-1", "admin", "admin")
    c.headers["Authorization"] = f"Bearer {token}"
    return c


def _fields(response):
    return response.json()["errors"]["fields"]


# ── Listing ───────────────────────────────────────────────
def test_list_returns_varieties_with_their_group(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.list_varieties.return_value = [VARIETY]
        response = client.get("/api/rules/varieties")

    assert response.status_code == 200
    variety = response.json()["data"]["varieties"][0]
    assert variety["name"] == "มันเสือ"
    assert variety["group_id"] == 1


def test_varieties_is_not_swallowed_by_the_group_route(client):
    """/api/rules/{group_id} is int-typed, but ordering must not rely on that."""
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.list_varieties.return_value = []
        response = client.get("/api/rules/varieties")

    assert response.status_code == 200
    mock_repo.get_group.assert_not_called()


# ── Batch create ──────────────────────────────────────────
def test_many_names_go_into_one_group(client):
    names = ["มันเสือ", "มันขาว", "มันแดง"]
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.group_ids_that_exist.return_value = {1}
        mock_repo.existing_variety_names.return_value = set()
        mock_repo.create_varieties.return_value = [1, 2, 3]
        mock_repo.get_varieties.return_value = [
            {**VARIETY, "id": i + 1, "name": n} for i, n in enumerate(names)
        ]
        response = client.post("/api/rules/varieties", json={
            "group_id": 1,
            "items": [{"name": n, "is_active": True} for n in names],
        })

    assert response.status_code == 201
    assert response.json()["data"]["created"] == 3
    group_id, items = mock_repo.create_varieties.call_args.args
    assert group_id == 1
    assert [i["name"] for i in items] == names


def test_names_are_stripped(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.group_ids_that_exist.return_value = {1}
        mock_repo.existing_variety_names.return_value = set()
        mock_repo.create_varieties.return_value = [1]
        mock_repo.get_varieties.return_value = [VARIETY]
        client.post("/api/rules/varieties", json={
            "group_id": 1, "items": [{"name": "  มันเสือ  "}],
        })

    assert mock_repo.create_varieties.call_args.args[1][0]["name"] == "มันเสือ"


def test_per_row_active_flags_are_kept(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.group_ids_that_exist.return_value = {1}
        mock_repo.existing_variety_names.return_value = set()
        mock_repo.create_varieties.return_value = [1, 2]
        mock_repo.get_varieties.return_value = [VARIETY]
        client.post("/api/rules/varieties", json={
            "group_id": 1,
            "items": [{"name": "a", "is_active": True},
                      {"name": "b", "is_active": False}],
        })

    items = mock_repo.create_varieties.call_args.args[1]
    assert [i["is_active"] for i in items] == [True, False]


# ── Indexed validation ────────────────────────────────────
def test_an_empty_name_names_its_row(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.group_ids_that_exist.return_value = {1}
        response = client.post("/api/rules/varieties", json={
            "group_id": 1,
            "items": [{"name": "มันเสือ"}, {"name": "   "}],
        })

    assert response.status_code == 422
    assert _fields(response) == [
        {"index": 1, "field": "name", "message": "กรุณากรอกชื่อชนิดมัน"}
    ]
    mock_repo.create_varieties.assert_not_called()


def test_a_name_duplicated_within_the_batch_points_at_the_first_row(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.group_ids_that_exist.return_value = {1}
        response = client.post("/api/rules/varieties", json={
            "group_id": 1,
            "items": [{"name": "มันเสือ"}, {"name": "มันขาว"}, {"name": "มันเสือ"}],
        })

    assert response.status_code == 422
    error = _fields(response)[0]
    assert error["index"] == 2
    assert "บรรทัดที่ 1" in error["message"]
    mock_repo.create_varieties.assert_not_called()


def test_batch_duplicates_are_matched_case_insensitively(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.group_ids_that_exist.return_value = {1}
        response = client.post("/api/rules/varieties", json={
            "group_id": 1, "items": [{"name": "Cassava"}, {"name": "cassava"}],
        })

    assert _fields(response)[0]["index"] == 1


def test_a_name_already_in_the_database_names_its_row(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.group_ids_that_exist.return_value = {1}
        mock_repo.existing_variety_names.return_value = {"มันขาว"}
        response = client.post("/api/rules/varieties", json={
            "group_id": 1, "items": [{"name": "มันเสือ"}, {"name": "มันขาว"}],
        })

    assert response.status_code == 422
    assert _fields(response) == [
        {"index": 1, "field": "name", "message": "ชื่อนี้มีอยู่แล้วในระบบ"}
    ]


def test_several_bad_rows_are_all_reported_in_order(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.group_ids_that_exist.return_value = {1}
        response = client.post("/api/rules/varieties", json={
            "group_id": 1,
            "items": [{"name": ""}, {"name": "ok"}, {"name": ""}],
        })

    assert [e["index"] for e in _fields(response)] == [0, 2]


def test_an_unknown_group_is_a_form_level_error(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.group_ids_that_exist.return_value = set()
        response = client.post("/api/rules/varieties", json={
            "group_id": 999, "items": [{"name": "มันเสือ"}],
        })

    assert response.status_code == 422
    assert _fields(response)[0] == {
        "index": None, "field": "group_id", "message": "ไม่พบกลุ่มนี้"
    }


def test_an_empty_batch_is_rejected(client):
    with patch("src.rules.router.repository"):
        response = client.post(
            "/api/rules/varieties", json={"group_id": 1, "items": []}
        )

    assert response.status_code == 422


def test_a_type_error_is_also_reported_by_index(client):
    """The shared handler turns pydantic's loc path into index + field."""
    with patch("src.rules.router.repository"):
        response = client.post("/api/rules/varieties", json={
            "group_id": 1, "items": [{"name": "ok"}, {"name": 5, "is_active": "yes"}],
        })

    assert response.status_code == 422
    assert all(e["index"] == 1 for e in _fields(response))
    assert {e["field"] for e in _fields(response)} <= {"name", "is_active"}


# ── Update ────────────────────────────────────────────────
def test_update_moves_a_variety_to_another_group(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.group_ids_that_exist.return_value = {2}
        mock_repo.existing_variety_names.return_value = set()
        mock_repo.update_variety.return_value = True
        mock_repo.get_variety.return_value = {**VARIETY, "group_id": 2}
        response = client.put("/api/rules/varieties/1", json={
            "group_id": 2, "name": "มันเสือ", "is_active": True,
        })

    assert response.status_code == 200
    assert mock_repo.update_variety.call_args.args[1]["group_id"] == 2


def test_update_may_keep_its_own_name(client):
    """The uniqueness check must exclude the row being edited."""
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.group_ids_that_exist.return_value = {1}
        mock_repo.existing_variety_names.return_value = set()
        mock_repo.update_variety.return_value = True
        mock_repo.get_variety.return_value = VARIETY
        client.put("/api/rules/varieties/1", json={
            "group_id": 1, "name": "มันเสือ", "is_active": True,
        })

    assert mock_repo.existing_variety_names.call_args.kwargs == {"exclude_id": 1}


def test_update_to_a_taken_name_is_reported_on_row_zero(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.group_ids_that_exist.return_value = {1}
        mock_repo.existing_variety_names.return_value = {"มันขาว"}
        response = client.put("/api/rules/varieties/1", json={
            "group_id": 1, "name": "มันขาว", "is_active": True,
        })

    assert response.status_code == 422
    assert _fields(response)[0]["index"] == 0


def test_update_missing_variety_is_404(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.group_ids_that_exist.return_value = {1}
        mock_repo.existing_variety_names.return_value = set()
        mock_repo.update_variety.return_value = False
        response = client.put("/api/rules/varieties/99", json={
            "group_id": 1, "name": "มันเสือ", "is_active": True,
        })

    assert response.status_code == 404


# ── Read / delete ─────────────────────────────────────────
def test_get_missing_variety_is_404(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.get_variety.return_value = None
        response = client.get("/api/rules/varieties/99")

    assert response.status_code == 404


def test_delete_removes_the_variety(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.delete_variety.return_value = True
        response = client.delete("/api/rules/varieties/1")

    assert response.status_code == 200
    assert response.json()["data"]["deleted"] == 1


def test_delete_missing_variety_is_404(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.delete_variety.return_value = False
        response = client.delete("/api/rules/varieties/99")

    assert response.status_code == 404
