# schemas/auth.py
from pydantic import BaseModel


class GenTokenRequest(BaseModel):
    username: str
    password: str
