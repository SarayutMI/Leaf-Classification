# src/classify/repository.py
"""Raw-SQL data access for classification logs and stored regions."""
from src.database import get_conn


def save_image_dataset(log_id: int, cdn_urls: dict[str, str]) -> None:
    """Save S3 CDN URLs for each region linked to a log entry."""
    conn = get_conn()
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
    conn = get_conn()
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
