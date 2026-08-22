# src/auth/service.py
"""Business logic for issuing API keys."""
from src.auth import repository, security
from src.auth.exceptions import InvalidCredentials


def get_or_create_api_key(username: str, password: str) -> str:
    """Return the caller's API key, registering a new user on first use.

    Raises InvalidCredentials when the user exists and the password is wrong.
    """
    user = repository.get_user_by_username(username)

    if user:
        if not security.verify_password(password, user["password_hash"]):
            raise InvalidCredentials
        token = repository.get_token_by_user_id(user["id"])
        return token["api_key"]

    password_hash = security.hash_password(password)
    user_id = repository.create_user(username, password_hash)
    api_key = security.generate_api_key()
    repository.create_token(user_id, api_key)
    return api_key
