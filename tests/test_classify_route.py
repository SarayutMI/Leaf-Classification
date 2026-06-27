import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse
from core.dependencies import require_api_key

app = FastAPI()


@app.post("/classify")
async def mock_classify(api_key: str = Depends(require_api_key)):
    return JSONResponse(status_code=200, content={"code": 200, "status": "success"})


client = TestClient(app, raise_server_exceptions=False)

VALID_KEY = "abc123def456abc123def456abc123de"


def test_classify_missing_api_key_returns_401():
    with patch("core.dependencies.database") as mock_db:
        response = client.post("/classify")
        assert response.status_code == 401
        assert response.json()["detail"] == "API key required"
        mock_db.log_api_call.assert_called_once()
        call_kwargs = mock_db.log_api_call.call_args.kwargs
        assert call_kwargs["http_status"] == 401
        assert call_kwargs["status"] == "error"


def test_classify_invalid_api_key_returns_401():
    with patch("core.dependencies.database") as mock_db:
        mock_db.get_token_by_api_key.return_value = None
        response = client.post("/classify", headers={"X-API-Key": "invalidkey"})
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid API key"
        mock_db.log_api_call.assert_called_once()
        call_kwargs = mock_db.log_api_call.call_args.kwargs
        assert call_kwargs["http_status"] == 401
        assert call_kwargs["api_key"] == "invalidkey"


def test_classify_valid_api_key_passes():
    with patch("core.dependencies.database") as mock_db:
        mock_db.get_token_by_api_key.return_value = {"api_key": VALID_KEY}
        response = client.post("/classify", headers={"X-API-Key": VALID_KEY})
        assert response.status_code == 200
        mock_db.log_api_call.assert_not_called()
