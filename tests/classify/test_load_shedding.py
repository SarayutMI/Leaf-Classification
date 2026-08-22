# tests/classify/test_load_shedding.py
"""The 429 guard that keeps queued callers from timing out."""
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.classify import router as classify_router
from src.classify.router import router

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)

VALID_KEY = "abc123def456abc123def456abc123de"


@pytest.fixture(autouse=True)
def reset_inflight():
    classify_router._inflight = 0
    yield
    classify_router._inflight = 0


def _post(jpeg_bytes):
    return client.post(
        "/api/classify",
        headers={"X-API-Key": VALID_KEY},
        files={"image": ("leaf.jpg", jpeg_bytes, "image/jpeg")},
    )


def test_rejects_with_429_once_the_limit_is_reached(jpeg_bytes):
    with patch("src.classify.router.repository") as mock_repo, \
         patch("src.auth.dependencies.auth_repository") as mock_tokens:
        mock_tokens.get_token_by_api_key.return_value = {"api_key": VALID_KEY}
        classify_router._inflight = classify_router.settings.MAX_CONCURRENT_CLASSIFY

        response = _post(jpeg_bytes)

        assert response.status_code == 429
        assert response.headers["Retry-After"] == str(classify_router.RETRY_AFTER_SECONDS)
        assert response.json()["errors"]["type"] == "RATE_LIMIT"

        mock_repo.log_api_call.assert_called_once()
        assert mock_repo.log_api_call.call_args.kwargs["http_status"] == 429


def test_inflight_is_released_after_a_successful_request(jpeg_bytes):
    with patch("src.classify.router.repository") as mock_repo, \
         patch("src.classify.router.service") as mock_service, \
         patch("src.classify.router.rules_service") as mock_rules, \
         patch("src.auth.dependencies.auth_repository") as mock_tokens:
        mock_tokens.get_token_by_api_key.return_value = {"api_key": VALID_KEY}
        mock_rules.match_group.return_value = []
        mock_service.slice_leaf.return_value = {
            "full": object(), "top": object(), "middle": object(), "bottom": object()
        }
        mock_service.predict_all.return_value = {
            "shape":  ("Ovate", 90.0),
            "apex":   ("Acute", 85.0),
            "base":   ("Cuneate", 80.0),
            "margin": ("Entire", 75.0),
        }
        mock_repo.log_api_call.return_value = 1

        assert _post(jpeg_bytes).status_code == 200
        assert classify_router._inflight == 0


def test_inflight_is_released_after_a_failure(jpeg_bytes):
    with patch("src.classify.router.repository"), \
         patch("src.classify.router.service") as mock_service, \
         patch("src.auth.dependencies.auth_repository") as mock_tokens:
        mock_tokens.get_token_by_api_key.return_value = {"api_key": VALID_KEY}
        mock_service.detect_leaf.side_effect = RuntimeError("boom")

        assert _post(jpeg_bytes).status_code == 500
        assert classify_router._inflight == 0
