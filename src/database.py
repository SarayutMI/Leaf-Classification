# src/database.py
import threading
import mysql.connector
from mysql.connector import pooling
from src.config import settings

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
                    host=settings.DB_HOST,
                    port=settings.DB_PORT,
                    user=settings.DB_USER,
                    password=settings.DB_PASSWORD,
                    database=settings.DB_NAME,
                    connection_timeout=10,
                    time_zone="+07:00",
                )
    return _pool


def get_conn():
    """Borrow a pooled connection. Callers must close() it (see the
    try/finally blocks in the feature repositories)."""
    return _get_pool().get_connection()


