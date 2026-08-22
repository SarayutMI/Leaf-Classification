import pytest
from src.auth.security import hash_password, verify_password, generate_api_key


def test_hash_password_returns_string():
    result = hash_password("secret123")
    assert isinstance(result, str)
    assert len(result) > 0


def test_hash_password_is_not_plain_text():
    result = hash_password("secret123")
    assert result != "secret123"


def test_verify_password_correct():
    hashed = hash_password("mypassword")
    assert verify_password("mypassword", hashed) is True


def test_verify_password_wrong():
    hashed = hash_password("mypassword")
    assert verify_password("wrongpassword", hashed) is False


def test_generate_api_key_is_32_chars():
    key = generate_api_key()
    assert len(key) == 32


def test_generate_api_key_is_unique():
    key1 = generate_api_key()
    key2 = generate_api_key()
    assert key1 != key2


def test_generate_api_key_is_hex():
    key = generate_api_key()
    int(key, 16)  # raises ValueError if not valid hex
