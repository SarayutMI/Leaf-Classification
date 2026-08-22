# tests/test_health_route.py
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.health import router as health_router
from src.health.router import router

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def clear_probe_cache():
    health_router._db_probe = (0.0, False)
    yield
    health_router._db_probe = (0.0, False)


@pytest.fixture
def loaded_models():
    with patch.object(health_router, "leaf_service") as svc:
        for name in ("yolo_model", "shape_model", "apex_model", "base_model", "margin_model"):
            setattr(svc, name, MagicMock())
        yield svc


def test_live_never_touches_the_database():
    with patch("src.health.router.get_conn") as mock_conn:
        response = client.get("/health/live")

        assert response.status_code == 200
        assert response.json()["services"] == {"api": "ok"}
        mock_conn.assert_not_called()


def test_successful_db_ping_is_cached(loaded_models):
    with patch("src.health.router.get_conn") as mock_conn:
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200

        # Three requests, one round trip to the database.
        assert mock_conn.call_count == 1


def test_failed_db_ping_is_not_cached(loaded_models):
    with patch("src.health.router.get_conn") as mock_conn:
        mock_conn.side_effect = RuntimeError("connection refused")

        first = client.get("/health")
        assert first.status_code == 503
        assert first.json()["services"]["database"] == "error"

        # A recovering database must be picked up on the very next probe.
        mock_conn.side_effect = None
        assert client.get("/health").status_code == 200
        assert mock_conn.call_count == 2


def test_unloaded_models_report_503():
    with patch("src.health.router.get_conn"), \
         patch.object(health_router, "leaf_service") as svc:
        svc.yolo_model = None
        for name in ("shape_model", "apex_model", "base_model", "margin_model"):
            setattr(svc, name, MagicMock())

        response = client.get("/health")

        assert response.status_code == 503
        assert response.json()["services"]["models"]["yolo"] == "not_loaded"
