# routes/auth.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from schemas.auth import GenTokenRequest
from core import database, security

router = APIRouter()


@router.post("/api/genToken")
async def gen_token(body: GenTokenRequest):
    try:
        user = database.get_user_by_username(body.username)

        if user:
            if not security.verify_password(body.password, user["password_hash"]):
                return JSONResponse(
                    status_code=401,
                    content={"code": 401, "status": "error", "message": "Invalid credentials"},
                )
            token = database.get_token_by_user_id(user["id"])
            api_key = token["api_key"]
        else:
            password_hash = security.hash_password(body.password)
            user_id = database.create_user(body.username, password_hash)
            api_key = security.generate_api_key()
            database.create_token(user_id, api_key)

        return JSONResponse(
            status_code=200,
            content={"code": 200, "status": "success", "data": {"api_key": api_key}},
        )

    except Exception:
        return JSONResponse(
            status_code=500,
            content={"code": 500, "status": "error", "message": "Database error"},
        )
