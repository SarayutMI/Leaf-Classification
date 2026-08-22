# src/auth/router.py
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.auth import repository as auth_repository, security, service
from src.auth.dependencies import NO_STORE, require_session
from src.auth.exceptions import InvalidCredentials
from src.auth.schemas import AdminLoginRequest, GenTokenRequest
from src.config import settings

router = APIRouter(prefix="/api", tags=["auth"])
logger = logging.getLogger(__name__)


def _invalid_credentials() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"code": 401, "status": "error", "message": "Invalid credentials"},
        headers=NO_STORE,
    )


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


@router.post("/admin/login")
async def admin_login(body: AdminLoginRequest):
    """Log in to the rule-base admin API.

    Credentials come from the platform's `users` table, not this service's
    classify_user (which still backs /api/genToken). Only an active admin gets
    a token.

    Every rejection returns the same 401 body: telling a caller which admin
    usernames exist, or which are disabled, is free reconnaissance.
    """
    try:
        user = auth_repository.get_admin_user_by_username(body.username)
    except Exception:
        logger.exception("Admin login failed to read user")
        return JSONResponse(
            status_code=500,
            content={"code": 500, "status": "error", "message": "Database error"},
        )

    if not user:
        return _invalid_credentials()

    if not security.verify_password(body.password, user["password"]):
        return _invalid_credentials()

    if user["role"] != "admin":
        logger.warning("Admin login refused for non-admin user %r", body.username)
        return _invalid_credentials()

    if not user["is_active"]:
        logger.warning("Admin login refused for disabled user %r", body.username)
        return _invalid_credentials()

    token = security.create_access_token(user["id"], user["username"], user["role"])
    return JSONResponse(
        status_code=200,
        content={
            "code": 200,
            "status": "success",
            "data": {
                "access_token": token,
                "token_type": "bearer",
                "expires_in": settings.JWT_EXPIRE_MINUTES * 60,
                "user": {
                    "id": str(user["id"]),
                    "username": user["username"],
                    "name": user["name"],
                    "role": user["role"],
                },
            },
        },
        headers=NO_STORE,
    )


@router.post("/admin/logout")
async def admin_logout():
    """No server-side session exists to end — the client discards its token.

    Kept as an endpoint so the page has one call to make and the API shape does
    not change. A token stays valid until it expires; there is no revocation.
    """
    return JSONResponse(
        status_code=200,
        content={
            "code": 200,
            "status": "success",
            "message": "Discard the access token on the client",
        },
        headers=NO_STORE,
    )


@router.get("/admin/me")
async def admin_me(session: dict = Depends(require_session)):
    return JSONResponse(
        status_code=200,
        content={
            "code": 200,
            "status": "success",
            "data": {
                "id": session.get("sub"),
                "username": session.get("username"),
                "role": session.get("role"),
            },
        },
        headers=NO_STORE,
    )
