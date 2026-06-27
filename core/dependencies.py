# core/dependencies.py
import logging
from fastapi import Header, HTTPException, Request
from core import database

logger = logging.getLogger(__name__)


async def require_api_key(
    request: Request,
    x_api_key: str = Header(default=None),
) -> str:
    ip = request.client.host if request.client else "unknown"

    if not x_api_key:
        try:
            database.log_api_call(
                api_key="",
                ip_address=ip,
                filename="unknown",
                http_status=401,
                status="error",
                error_message="API key required",
            )
        except Exception:
            logger.exception("Failed to write API log (missing key)")
        raise HTTPException(status_code=401, detail="API key required")

    token = database.get_token_by_api_key(x_api_key)
    if not token:
        try:
            database.log_api_call(
                api_key=x_api_key,
                ip_address=ip,
                filename="unknown",
                http_status=401,
                status="error",
                error_message="Invalid API key",
            )
        except Exception:
            logger.exception("Failed to write API log (invalid key)")
        raise HTTPException(status_code=401, detail="Invalid API key")

    return x_api_key
