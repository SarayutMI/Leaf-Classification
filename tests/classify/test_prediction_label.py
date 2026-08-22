# tests/classify/test_prediction_label.py
"""/api/classify labels the leaf with a rule-base group."""
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.classify.router import router

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)

VALID_KEY = "abc123def456abc123def456abc123de"

PREDICTIONS = {
    "shape":  ("Cordate", 90.0),
    "apex":   ("Acute", 80.0),
    "base":   ("Auriculate", 70.0),
    "margin": ("Entire", 60.0),
}


def _classify(jpeg_bytes, match_group_result=None, match_group_error=None):
    with patch("src.classify.router.repository") as mock_repo, \
         patch("src.classify.router.service.detect_leaf") as mock_detect, \
         patch("src.classify.router.service.slice_leaf") as mock_slice, \
         patch("src.classify.router.service.predict_all") as mock_predict, \
         patch("src.classify.router.rules_service") as mock_rules, \
         patch("src.auth.dependencies.auth_repository") as mock_tokens:
        mock_tokens.get_token_by_api_key.return_value = {"api_key": VALID_KEY}
        mock_detect.return_value = object()
        mock_slice.return_value = {
            "full": object(), "top": object(), "middle": object(), "bottom": object()
        }
        mock_predict.return_value = PREDICTIONS
        mock_repo.log_api_call.return_value = 1
        if match_group_error is not None:
            mock_rules.match_group.side_effect = match_group_error
        else:
            mock_rules.match_group.return_value = match_group_result or []

        response = client.post(
            "/api/classify",
            headers={"X-API-Key": VALID_KEY},
            files={"image": ("leaf.jpg", jpeg_bytes, "image/jpeg")},
        )
    return response, mock_rules, mock_repo


def test_best_group_becomes_the_prediction_label(jpeg_bytes):
    response, _, mock_repo = _classify(jpeg_bytes, [
        {"id": 1, "code": "G1", "name": "กลุ่มใบหัวใจ", "matched": 4, "score": 3.5},
        {"id": 2, "code": "G2", "name": "กลุ่มใบไข่", "matched": 2, "score": 1.0},
    ])

    assert response.status_code == 200
    prediction = response.json()["data"]["prediction"]
    # group_id and code let the caller fetch the group's varieties.
    assert prediction == {
        "group_id": 1, "code": "G1", "label": "กลุ่มใบหัวใจ", "confidence": 75.0,
    }
    assert mock_repo.log_api_call.call_args.kwargs["prediction_label"] == "กลุ่มใบหัวใจ"


def test_probabilities_are_passed_as_fractions_not_percentages(jpeg_bytes):
    """predict_class returns 0-100; match_group's conf_th works on 0-1."""
    _, mock_rules, _ = _classify(jpeg_bytes, [])

    kwargs = mock_rules.match_group.call_args.kwargs
    assert kwargs["probs"] == {
        "shape": 0.9, "apex": 0.8, "base": 0.7, "margin": 0.6,
    }


def test_empty_rule_base_yields_a_null_group(jpeg_bytes):
    response, _, _ = _classify(jpeg_bytes, [])

    assert response.status_code == 200
    prediction = response.json()["data"]["prediction"]
    assert prediction["label"] is None
    assert prediction["group_id"] is None
    assert prediction["code"] is None


def test_rule_failure_does_not_break_the_classification(jpeg_bytes):
    """A bad rule or a database blip must not 500 a successful prediction."""
    response, _, _ = _classify(jpeg_bytes, match_group_error=RuntimeError("db down"))

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["prediction"]["label"] is None
    assert data["prediction"]["group_id"] is None
    # The trait predictions still come back.
    assert data["shape"] == "Cordate"
