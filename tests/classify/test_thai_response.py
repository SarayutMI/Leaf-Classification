# tests/classify/test_thai_response.py
"""/api/classify returns Thai names next to the English ones."""
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.classify.router import router

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)

VALID_KEY = "abc123def456abc123def456abc123de"


def _classify(jpeg_bytes, predictions):
    with patch("src.classify.router.repository") as mock_repo, \
         patch("src.classify.router.service.detect_leaf") as mock_detect, \
         patch("src.classify.router.service.slice_leaf") as mock_slice, \
         patch("src.classify.router.service.predict_all") as mock_predict, \
         patch("src.auth.dependencies.auth_repository") as mock_tokens:
        mock_tokens.get_token_by_api_key.return_value = {"api_key": VALID_KEY}
        mock_slice.return_value = {
            "full": object(), "top": object(), "middle": object(), "bottom": object()
        }
        mock_predict.return_value = predictions
        mock_repo.log_api_call.return_value = 1

        response = client.post(
            "/api/classify",
            headers={"X-API-Key": VALID_KEY},
            files={"image": ("leaf.jpg", jpeg_bytes, "image/jpeg")},
        )
    return response


def test_thai_names_sit_next_to_the_english_ones(jpeg_bytes):
    response = _classify(jpeg_bytes, {
        "shape":  ("Cordate", 90.0),
        "apex":   ("Acute", 85.0),
        "base":   ("Cuneate", 80.0),
        "margin": ("Entire", 75.0),
    })

    assert response.status_code == 200
    data = response.json()["data"]

    # English keys unchanged — existing clients must not break.
    assert data["shape"] == "Cordate"
    assert data["apex"] == "Acute"
    assert data["base"] == "Cuneate"
    assert data["margin"] == "Entire"

    assert data["shape_th"] == "รูปหัวใจ"
    assert data["apex_th"] == "แหลม"
    assert data["base_th"] == "รูปลิ่ม"
    assert data["margin_th"] == "เรียบ"


def test_caudate_is_translated_per_region(jpeg_bytes):
    """The same English term, two different Thai names — apex vs base."""
    response = _classify(jpeg_bytes, {
        "shape":  ("Ovate", 90.0),
        "apex":   ("Caudate", 85.0),
        "base":   ("Caudate", 80.0),
        "margin": ("Crenate", 75.0),
    })

    data = response.json()["data"]
    assert data["apex_th"] == "ยาวคล้ายหาง"
    assert data["base_th"] == "รูปหัวใจ"


def test_response_is_valid_utf8_json(jpeg_bytes):
    response = _classify(jpeg_bytes, {
        "shape":  ("Sagittate", 90.0),
        "apex":   ("Obtuse", 85.0),
        "base":   ("Auriculate", 80.0),
        "margin": ("Crenate", 75.0),
    })

    # Decodes without mangling, and the Thai survives the round trip.
    assert response.json()["data"]["base_th"] == "รูปติ่งหู"
