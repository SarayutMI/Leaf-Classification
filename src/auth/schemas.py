# src/auth/schemas.py
from pydantic import BaseModel


class GenTokenRequest(BaseModel):
    username: str
    password: str
