# src/auth/router.py
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.auth import service
from src.auth.exceptions import InvalidCredentials
from src.auth.schemas import GenTokenRequest

router = APIRouter(prefix="/api", tags=["auth"])
logger = logging.getLogger(__name__)


@router.post("/genToken")
async def gen_token(body: GenTokenRequest):
    try:
        api_key = service.get_or_create_api_key(body.username, body.password)
    except InvalidCredentials:
        return JSONResponse(
            status_code=401,
            content={"code": 401, "status": "error", "message": "Invalid credentials"},
        )
    except Exception:
        logger.exception("Failed to issue API key")
        return JSONResponse(
            status_code=500,
            content={"code": 500, "status": "error", "message": "Database error"},
        )

    return JSONResponse(
        status_code=200,
        content={"code": 200, "status": "success", "data": {"api_key": api_key}},
    )
