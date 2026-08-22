# tests/rules/test_router.py
"""CRUD routes for the rule base. The repository is mocked, so no database."""
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.auth import security
from src.config import settings
from src.rules.router import router

app = FastAPI()
app.include_router(router)


GROUP = {
    "id": 1, "code": "G1", "name": "กลุ่มใบหัวใจ", "is_active": True,
    "shape": "Cordate", "apex": "Acute", "base": "Auriculate", "margin": "Entire",
}

PAYLOAD = {
    "code": "G1", "name": "กลุ่มใบหัวใจ", "is_active": True,
    "shape": "Cordate", "apex": "Acute", "base": "Auriculate", "margin": "Entire",
}


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setattr(settings, "JWT_SECRET", "test-secret-not-a-real-key")


@pytest.fixture
def client():
    """Authenticated client — carries a valid Bearer token."""
    c = TestClient(app, raise_server_exceptions=False)
    token = security.create_access_token("uuid-1", "admin", "admin")
    c.headers["Authorization"] = f"Bearer {token}"
    return c


@pytest.fixture
def anonymous():
    return TestClient(app, raise_server_exceptions=False)


def _declared_routes():
    """Every route on the rules router, read off the router itself.

    Enumerated rather than hand-listed so a route added later is covered here
    without anyone remembering to extend a list.
    """
    for route in router.routes:
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            yield method, route.path


@pytest.mark.parametrize("method,path", list(_declared_routes()))
def test_every_route_requires_a_session(anonymous, method, path):
    # request() rather than get()/post(): only the body-carrying verbs accept
    # json=, and the auth check has to happen before body validation anyway.
    response = anonymous.request(
        method, path.replace("{variety_id}", "1"), json={}
    )
    assert response.status_code == 401


def test_list_returns_groups_with_a_variety_count(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.list_groups.return_value = [dict(GROUP)]
        mock_repo.variety_counts_by_group.return_value = {1: 2}
        response = client.get("/api/rules")

    assert response.status_code == 200
    group = response.json()["data"]["groups"][0]
    assert group["code"] == "G1"
    assert group["variety_count"] == 2


def test_a_group_with_no_varieties_counts_zero(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.list_groups.return_value = [dict(GROUP)]
        mock_repo.variety_counts_by_group.return_value = {}
        response = client.get("/api/rules")

    assert response.json()["data"]["groups"][0]["variety_count"] == 0


def test_vocab_lists_the_configured_classes(client):
    response = client.get("/api/rules/vocab")
    assert response.status_code == 200
    vocab = response.json()["data"]["vocab"]
    assert vocab["shape"] == settings.SHAPE_CLASSES
    assert vocab["margin"] == settings.MARGIN_CLASSES


def test_create_stores_the_group(client):
    with patch("src.rules.router.repository") as mock_repo, \
         patch("src.rules.router.service") as mock_service:
        mock_repo.code_exists.return_value = False
        mock_repo.combination_exists.return_value = False
        mock_repo.create_group.return_value = 1
        mock_repo.get_group.return_value = GROUP
        response = client.post("/api/rules", json=PAYLOAD)

    assert response.status_code == 201
    assert mock_repo.create_group.call_args.args[0]["shape"] == "Cordate"
    mock_service.invalidate_cache.assert_called_once()


def test_a_lowercase_class_is_stored_in_config_casing(client):
    """The models emit "Cordate"; the rule must match it without a fuzzy compare."""
    with patch("src.rules.router.repository") as mock_repo, \
         patch("src.rules.router.service"):
        mock_repo.code_exists.return_value = False
        mock_repo.combination_exists.return_value = False
        mock_repo.create_group.return_value = 1
        mock_repo.get_group.return_value = GROUP
        client.post("/api/rules", json={**PAYLOAD, "shape": "cordate"})

    assert mock_repo.create_group.call_args.args[0]["shape"] == "Cordate"


def test_duplicate_combination_is_409(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.code_exists.return_value = False
        mock_repo.combination_exists.return_value = True
        response = client.post("/api/rules", json=PAYLOAD)

    assert response.status_code == 409
    mock_repo.create_group.assert_not_called()


def test_duplicate_code_is_409(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.code_exists.return_value = True
        response = client.post("/api/rules", json=PAYLOAD)

    assert response.status_code == 409
    mock_repo.create_group.assert_not_called()


def test_unknown_trait_value_is_rejected(client):
    with patch("src.rules.router.repository"):
        response = client.post("/api/rules", json={**PAYLOAD, "shape": "Triangular"})

    assert response.status_code == 422


def test_wildcard_is_no_longer_accepted(client):
    with patch("src.rules.router.repository"):
        response = client.post("/api/rules", json={**PAYLOAD, "margin": "*"})

    assert response.status_code == 422


def test_a_class_from_the_wrong_trait_is_rejected(client):
    """"Crenate" is a margin, never a shape."""
    with patch("src.rules.router.repository"):
        response = client.post("/api/rules", json={**PAYLOAD, "shape": "Crenate"})

    assert response.status_code == 422


def test_every_trait_is_required(client):
    payload = {k: v for k, v in PAYLOAD.items() if k != "margin"}
    with patch("src.rules.router.repository"):
        response = client.post("/api/rules", json=payload)

    assert response.status_code == 422


def test_update_missing_group_is_404(client):
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.code_exists.return_value = False
        mock_repo.combination_exists.return_value = False
        mock_repo.update_group.return_value = False
        response = client.put("/api/rules/99", json=PAYLOAD)

    assert response.status_code == 404


def test_update_invalidates_the_cache(client):
    with patch("src.rules.router.repository") as mock_repo, \
         patch("src.rules.router.service") as mock_service:
        mock_repo.code_exists.return_value = False
        mock_repo.combination_exists.return_value = False
        mock_repo.update_group.return_value = True
        mock_repo.get_group.return_value = GROUP
        response = client.put("/api/rules/1", json=PAYLOAD)

    assert response.status_code == 200
    mock_service.invalidate_cache.assert_called_once()


def test_delete_missing_group_is_404(client):
    with patch("src.rules.router.repository") as mock_repo, \
         patch("src.rules.router.service") as mock_service:
        mock_repo.delete_group.return_value = False
        response = client.delete("/api/rules/99")

    assert response.status_code == 404
    mock_service.invalidate_cache.assert_not_called()


def test_delete_invalidates_the_cache(client):
    with patch("src.rules.router.repository") as mock_repo, \
         patch("src.rules.router.service") as mock_service:
        mock_repo.delete_group.return_value = True
        response = client.delete("/api/rules/1")

    assert response.status_code == 200
    mock_service.invalidate_cache.assert_called_once()


def test_test_endpoint_returns_scored_groups(client):
    with patch("src.rules.router.service") as mock_service:
        mock_service.match_group.return_value = [
            {"code": "G1", "name": "กลุ่มใบหัวใจ", "matched": 4, "score": 3.5}
        ]
        response = client.post("/api/rules/test", json={
            "traits": {"shape": "Cordate"},
            "probs": {"shape": 0.9},
        })

    assert response.status_code == 200
    assert response.json()["data"]["results"][0]["code"] == "G1"


def test_responses_are_not_cacheable(client):
    """A browser must not replay an authenticated 200 after logout."""
    with patch("src.rules.router.repository") as mock_repo:
        mock_repo.list_groups.return_value = [dict(GROUP)]
        mock_repo.variety_counts_by_group.return_value = {}
        response = client.get("/api/rules")

    assert response.status_code == 200
    assert "no-store" in response.headers["cache-control"]


def test_unauthenticated_responses_are_not_cacheable(anonymous):
    response = anonymous.get("/api/rules/vocab")

    assert response.status_code == 401
    assert "no-store" in response.headers["cache-control"]
