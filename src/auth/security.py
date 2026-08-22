# src/auth/security.py
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from src.config import settings


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def generate_api_key() -> str:
    return uuid.uuid4().hex


def create_access_token(user_id: str, username: str, role: str) -> str:
    """Sign a short-lived admin token.

    Carried as `Authorization: Bearer` by the admin page and by any other
    service that needs these APIs. `sub` is the users.id UUID.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    """Return the token payload, or None if it is expired, tampered with or
    otherwise unusable. Callers turn None into a 401."""
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
