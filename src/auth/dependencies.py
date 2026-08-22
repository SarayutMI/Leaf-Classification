# src/auth/dependencies.py
import logging
from fastapi import Header, HTTPException, Request, Security
from fastapi.security import APIKeyCookie
from src.auth import repository as auth_repository, security
from src.classify import repository as classify_repository
from src.config import settings

logger = logging.getLogger(__name__)

# Declared as a security scheme so /docs marks the admin routes as protected
# and names the cookie, instead of just 401-ing with no explanation.
# auto_error=False: the 401 below carries our own message.
# A 401 is as cacheable as a 200 to a browser, and it is raised from the
# dependency — outside the routes' own response helpers — so it carries its
# own no-store.
NO_STORE = {"Cache-Control": "no-store, private", "Pragma": "no-cache"}

session_cookie = APIKeyCookie(
    name=settings.SESSION_COOKIE_NAME,
    scheme_name="adminSession",
    description="Session cookie set by POST /api/admin/login.",
    auto_error=False,
)


async def require_api_key(
    request: Request,
    x_api_key: str = Header(default=None),
) -> str:
    ip = request.client.host if request.client else "unknown"

    if not x_api_key:
        try:
            classify_repository.log_api_call(
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

    token = auth_repository.get_token_by_api_key(x_api_key)
    if not token:
        try:
            classify_repository.log_api_call(
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


async def require_session(token: str = Security(session_cookie)) -> dict:
    """Authenticate the admin CRUD page from its HttpOnly session cookie.

    Unlike require_api_key this writes nothing to classify_api_logs: those
    rows are keyed by api_key and admin traffic would pollute the dataset.
    """
    if not token:
        raise HTTPException(
            status_code=401, detail="Not authenticated", headers=NO_STORE
        )

    payload = security.decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=401, detail="Session expired or invalid", headers=NO_STORE
        )

    return payload
