# src/auth/exceptions.py
class AuthError(Exception):
    """Base class for authentication failures raised by the auth service."""


class InvalidCredentials(AuthError):
    """Username exists but the supplied password does not match."""
