# src/auth/router.py
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.auth import repository as auth_repository, security, service
from src.auth.dependencies import require_session
from src.auth.exceptions import InvalidCredentials
from src.auth.schemas import AdminLoginRequest, GenTokenRequest
from src.config import settings

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


@router.post("/admin/login")
async def admin_login(body: AdminLoginRequest):
    """Log in to the rule-base admin page.

    Deliberately does NOT go through service.get_or_create_api_key, which
    registers unknown usernames on first use — acceptable when handing out an
    API key, not acceptable for an admin login.
    """
    try:
        user = auth_repository.get_user_by_username(body.username)
    except Exception:
        logger.exception("Admin login failed to read user")
        return JSONResponse(
            status_code=500,
            content={"code": 500, "status": "error", "message": "Database error"},
        )

    if not user or not security.verify_password(body.password, user["password_hash"]):
        return JSONResponse(
            status_code=401,
            content={"code": 401, "status": "error", "message": "Invalid credentials"},
        )

    token = security.create_access_token(user["id"], user["username"])
    response = JSONResponse(
        status_code=200,
        content={
            "code": 200,
            "status": "success",
            "data": {"username": user["username"]},
        },
    )
    response.set_cookie(
        key=settings.SESSION_COOKIE_NAME,
        value=token,
        max_age=settings.JWT_EXPIRE_MINUTES * 60,
        httponly=True,
        samesite="lax",
        secure=settings.COOKIE_SECURE,
        path="/",
    )
    return response


@router.post("/admin/logout")
async def admin_logout():
    response = JSONResponse(
        status_code=200,
        content={"code": 200, "status": "success", "message": "Logged out"},
    )
    response.delete_cookie(key=settings.SESSION_COOKIE_NAME, path="/")
    return response


@router.get("/admin/me")
async def admin_me(session: dict = Depends(require_session)):
    return JSONResponse(
        status_code=200,
        content={
            "code": 200,
            "status": "success",
            "data": {"username": session.get("username")},
        },
        # Never let a browser replay this from cache after logout.
        headers={"Cache-Control": "no-store, private", "Pragma": "no-cache"},
    )
