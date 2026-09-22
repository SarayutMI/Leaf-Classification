# tests/classify/test_decision.py
"""/api/classify returns traits only; /api/decision maps traits to a group."""
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.classify.router import router
from src.main import validation_error_handler
from fastapi.exceptions import RequestValidationError

app = FastAPI()
app.include_router(router)
app.add_exception_handler(RequestValidationError, validation_error_handler)
client = TestClient(app, raise_server_exceptions=False)

VALID_KEY = "abc123def456abc123def456abc123de"

PREDICTIONS = {
    "shape":  ("Cordate", 90.0),
    "apex":   ("Acute", 80.0),
    "base":   ("Auriculate", 70.0),
    "margin": ("Entire", 60.0),
}

TRAITS = {"shape": "Cordate", "apex": "Acute", "base": "Auriculate", "margin": "Entire"}
GROUP = {"id": 1, "code": "G1", "name": "กลุ่มใบหัวใจ", "matched": 4.0, "score": 4.0,
         "hits": ["shape", "apex", "base", "margin"]}


def test_classify_returns_traits_and_id_without_a_group(jpeg_bytes):
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
        mock_repo.log_api_call.return_value = 42

        response = client.post(
            "/api/classify",
            headers={"X-API-Key": VALID_KEY},
            files={"image": ("leaf.jpg", jpeg_bytes, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert "prediction" not in data
    assert data["classify_id"] == 42
    assert data["confidence"] == {
        "shape": 90.0, "apex": 80.0, "base": 70.0, "margin": 60.0, "overall": 75.0,
    }
    mock_rules.match_group.assert_not_called()
    assert mock_repo.log_api_call.call_args.kwargs.get("prediction_label") is None


def _decide(body, match_group_result=None, match_group_error=None, row_found=True):
    with patch("src.classify.router.repository") as mock_repo, \
         patch("src.classify.router.rules_service.match_group") as mock_match, \
         patch("src.auth.dependencies.auth_repository") as mock_tokens:
        mock_tokens.get_token_by_api_key.return_value = {"api_key": VALID_KEY}
        if match_group_error is not None:
            mock_match.side_effect = match_group_error
        else:
            mock_match.return_value = match_group_result if match_group_result is not None else [GROUP]
        mock_repo.update_prediction_label.return_value = row_found

        response = client.post("/api/decision", headers={"X-API-Key": VALID_KEY}, json=body)
    return response, mock_match, mock_repo


def test_decision_returns_best_group_and_updates_the_log():
    response, mock_match, mock_repo = _decide({"classify_id": 42, **TRAITS})

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["classify_id"] == 42
    assert data["prediction"] == {
        "group_id": 1, "code": "G1", "label": "กลุ่มใบหัวใจ", "matched": 4.0, "confidence": 100.0,
    }

    kwargs = mock_match.call_args.kwargs
    assert kwargs["traits"] == TRAITS
    # User-supplied labels are certain; none may fall under conf_th.
    assert kwargs["probs"] == {k: 1.0 for k in TRAITS}
    mock_repo.update_prediction_label.assert_called_once_with(42, VALID_KEY, "กลุ่มใบหัวใจ", 100.0)


def test_decision_without_classify_id_touches_no_log():
    response, _, mock_repo = _decide(TRAITS)

    assert response.status_code == 200
    assert response.json()["data"]["classify_id"] is None
    mock_repo.update_prediction_label.assert_not_called()


def test_labels_are_case_insensitive_and_normalized():
    _, mock_match, _ = _decide({"shape": "cordate", "apex": " ACUTE", "base": "auriculate", "margin": "entire"})

    assert mock_match.call_args.kwargs["traits"] == TRAITS


def test_non_ml_vocabulary_classes_are_accepted():
    """Traits keyed in by hand may name any class in leaf_traits.php, not just
    the four per trait the models predict."""
    hand = {"shape": "Reniform", "apex": "Mucronate", "base": "Peltate", "margin": "Divided"}
    response, mock_match, _ = _decide(hand)

    assert response.status_code == 200
    assert mock_match.call_args.kwargs["traits"] == hand


def test_unknown_label_is_a_422():
    response, mock_match, _ = _decide({**TRAITS, "shape": "Round"})

    assert response.status_code == 422
    fields = response.json()["errors"]["fields"]
    assert fields[0]["field"] == "shape"
    mock_match.assert_not_called()


def test_missing_trait_is_a_422():
    body = dict(TRAITS)
    del body["margin"]
    response, _, _ = _decide(body)

    assert response.status_code == 422


def test_foreign_or_unknown_classify_id_is_a_404():
    response, _, _ = _decide({"classify_id": 99, **TRAITS}, row_found=False)

    assert response.status_code == 404
    assert response.json()["errors"]["type"] == "NOT_FOUND"


def test_rule_failure_yields_a_null_group():
    """A bad rule or a database blip must not 500 the decision."""
    response, _, mock_repo = _decide({"classify_id": 42, **TRAITS}, match_group_error=RuntimeError("db down"))

    assert response.status_code == 200
    prediction = response.json()["data"]["prediction"]
    assert prediction == {"group_id": None, "code": None, "label": None, "matched": None, "confidence": None}
    mock_repo.update_prediction_label.assert_called_once_with(42, VALID_KEY, None, None)


def test_empty_rule_base_yields_a_null_group():
    response, _, _ = _decide(TRAITS, match_group_result=[])

    assert response.status_code == 200
    assert response.json()["data"]["prediction"]["label"] is None


def test_final_confidence_averages_classify_confidence_over_agreeing_traits():
    three_of_four = {**GROUP, "matched": 3.0, "hits": ["shape", "apex", "base"]}
    confidence = {"shape": 90.0, "apex": 80.0, "base": 70.0, "margin": 60.0}
    response, mock_match, _ = _decide({**TRAITS, "confidence": confidence}, match_group_result=[three_of_four])

    assert response.status_code == 200
    # margin disagrees with the group, so it contributes 0: (90+80+70+0)/4
    assert response.json()["data"]["prediction"]["confidence"] == 60.0
    # Ranking still treats the user's labels as certain.
    assert mock_match.call_args.kwargs["probs"] == {k: 1.0 for k in TRAITS}


def test_confidence_out_of_range_is_a_422():
    confidence = {"shape": 150.0, "apex": 80.0, "base": 70.0, "margin": 60.0}
    response, _, _ = _decide({**TRAITS, "confidence": confidence})

    assert response.status_code == 422
