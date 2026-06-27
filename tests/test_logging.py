import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from routes.classify import router

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)

VALID_KEY = "abc123def456abc123def456abc123de"


def _make_image_bytes():
    import cv2
    import numpy as np
    img = __import__('numpy').zeros((100, 100, 3), dtype=__import__('numpy').uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_success_logs_classification_results():
    with patch("routes.classify.database") as mock_db, \
         patch("routes.classify.leaf_svc") as mock_leaf, \
         patch("core.dependencies.database") as mock_dep_db:

        mock_dep_db.get_token_by_api_key.return_value = {"api_key": VALID_KEY}
        mock_leaf.detect_leaf.return_value = MagicMock()
        mock_leaf.slice_leaf.return_value = {
            "full": MagicMock(), "top": MagicMock(),
            "middle": MagicMock(), "bottom": MagicMock()
        }
        mock_leaf.predict_class.side_effect = [
            ("Ovate", 90.0), ("Acute", 85.0), ("Cuneate", 80.0), ("Entire", 75.0)
        ]
        mock_leaf.shape_model = MagicMock()
        mock_leaf.apex_model = MagicMock()
        mock_leaf.base_model = MagicMock()
        mock_leaf.margin_model = MagicMock()

        image_bytes = _make_image_bytes()
        response = client.post(
            "/classify",
            headers={"X-API-Key": VALID_KEY},
            files={"image": ("leaf.jpg", image_bytes, "image/jpeg")},
        )

        assert response.status_code == 200
        mock_db.log_api_call.assert_called_once()
        call_kwargs = mock_db.log_api_call.call_args.kwargs
        assert call_kwargs["http_status"] == 200
        assert call_kwargs["status"] == "success"
        assert call_kwargs["shape"] == "Ovate"
        assert call_kwargs["apex"] == "Acute"
        assert call_kwargs["error_message"] is None


def test_invalid_image_logs_error():
    with patch("routes.classify.database") as mock_db, \
         patch("core.dependencies.database") as mock_dep_db:

        mock_dep_db.get_token_by_api_key.return_value = {"api_key": VALID_KEY}

        response = client.post(
            "/classify",
            headers={"X-API-Key": VALID_KEY},
            files={"image": ("bad.jpg", b"not-an-image", "image/jpeg")},
        )

        assert response.status_code == 500
        mock_db.log_api_call.assert_called_once()
        call_kwargs = mock_db.log_api_call.call_args.kwargs
        assert call_kwargs["http_status"] == 500
        assert call_kwargs["status"] == "error"
        assert call_kwargs["shape"] is None


def test_no_leaf_detected_logs_error():
    with patch("routes.classify.database") as mock_db, \
         patch("routes.classify.leaf_svc") as mock_leaf, \
         patch("core.dependencies.database") as mock_dep_db:

        mock_dep_db.get_token_by_api_key.return_value = {"api_key": VALID_KEY}
        mock_leaf.detect_leaf.return_value = None

        image_bytes = _make_image_bytes()
        response = client.post(
            "/classify",
            headers={"X-API-Key": VALID_KEY},
            files={"image": ("leaf.jpg", image_bytes, "image/jpeg")},
        )

        assert response.status_code == 500
        mock_db.log_api_call.assert_called_once()
        call_kwargs = mock_db.log_api_call.call_args.kwargs
        assert call_kwargs["http_status"] == 500
        assert call_kwargs["error_message"] == "No leaf found"


def test_exception_response_does_not_expose_error_detail():
    with patch("routes.classify.database") as mock_db, \
         patch("routes.classify.leaf_svc") as mock_leaf, \
         patch("core.dependencies.database") as mock_dep_db:

        mock_dep_db.get_token_by_api_key.return_value = {"api_key": VALID_KEY}
        mock_leaf.detect_leaf.side_effect = RuntimeError("internal model path /secret/path.keras")

        image_bytes = _make_image_bytes()
        response = client.post(
            "/classify",
            headers={"X-API-Key": VALID_KEY},
            files={"image": ("leaf.jpg", image_bytes, "image/jpeg")},
        )

        assert response.status_code == 500
        body = response.json()
        assert "/secret/path.keras" not in body["message"]
        assert "/secret/path.keras" not in body["errors"]["details"]
        assert body["message"] == "An internal error occurred. Please try again."
