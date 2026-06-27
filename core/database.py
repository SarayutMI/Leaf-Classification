# core/database.py
import threading
import mysql.connector
from mysql.connector import pooling
import config

_pool = None
_pool_lock = threading.Lock()


def _get_pool() -> pooling.MySQLConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = pooling.MySQLConnectionPool(
                    pool_name="leaf_pool",
                    pool_size=10,
                    pool_reset_session=True,
                    host=config.DB_HOST,
                    port=config.DB_PORT,
                    user=config.DB_USER,
                    password=config.DB_PASSWORD,
                    database=config.DB_NAME,
                    connection_timeout=10,
                    time_zone="+07:00",
                )
    return _pool


def _get_conn():
    return _get_pool().get_connection()


def get_user_by_username(username: str) -> dict | None:
    conn = _get_conn()
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
    conn = _get_conn()
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
    conn = _get_conn()
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
    conn = _get_conn()
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
    conn = _get_conn()
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


def save_image_dataset(log_id: int, cdn_urls: dict[str, str]) -> None:
    """Save S3 CDN URLs for each region linked to a log entry."""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO classify_image_dataset (log_id, region, cdn_url) VALUES (%s, %s, %s)",
            [(log_id, region, url) for region, url in cdn_urls.items()],
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def log_api_call(
    api_key: str,
    ip_address: str,
    filename: str,
    http_status: int,
    status: str,
    error_message: str | None = None,
    shape: str | None = None,
    apex: str | None = None,
    base: str | None = None,
    margin: str | None = None,
    confidence: float | None = None,
    duration: float | None = None,
    prediction_label: str | None = None,
) -> int:
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO classify_api_logs
                (api_key, ip_address, filename, http_status, status,
                 error_message, shape, apex, base, margin, confidence, duration, prediction_label)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (api_key, ip_address, filename, http_status, status,
             error_message, shape, apex, base, margin, confidence, duration, prediction_label),
        )
        conn.commit()
        log_id = cursor.lastrowid
        cursor.close()
        return log_id
    finally:
        conn.close()
