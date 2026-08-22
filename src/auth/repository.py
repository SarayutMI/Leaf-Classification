# src/auth/repository.py
"""Raw-SQL data access for users and API tokens."""
from src.database import get_conn


def get_user_by_username(username: str) -> dict | None:
    conn = get_conn()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, username, password_hash FROM classify_user WHERE username = %s",
            (username,),
        )
        user = cursor.fetchone()
        cursor.close()
        return user
    finally:
        conn.close()


def create_user(username: str, password_hash: str) -> int:
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO classify_user (username, password_hash) VALUES (%s, %s)",
            (username, password_hash),
        )
        conn.commit()
        user_id = cursor.lastrowid
        cursor.close()
        return user_id
    finally:
        conn.close()


def get_token_by_user_id(user_id: int) -> dict | None:
    conn = get_conn()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, user_id, api_key FROM classify_token WHERE user_id = %s",
            (user_id,),
        )
        token = cursor.fetchone()
        cursor.close()
        return token
    finally:
        conn.close()


def get_token_by_api_key(api_key: str) -> dict | None:
    conn = get_conn()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, user_id, api_key FROM classify_token WHERE api_key = %s",
            (api_key,),
        )
        token = cursor.fetchone()
        cursor.close()
        return token
    finally:
        conn.close()


def create_token(user_id: int, api_key: str) -> None:
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO classify_token (user_id, api_key) VALUES (%s, %s)",
            (user_id, api_key),
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()
