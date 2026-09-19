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


def update_prediction_label(
    log_id: int, api_key: str, label: str | None, final_confidence: float | None = None,
) -> bool:
    """Record the decided group and its final confidence on a classify log row.
    `confidence` keeps the model's value. False when the row does not exist or
    belongs to another API key."""
    conn = get_conn()
    try:
        cursor = conn.cursor()
        # Match on id alone first: MySQL reports 0 affected rows for an UPDATE
        # that changes nothing, so a repeated decision would look like a 404.
        cursor.execute(
            "SELECT 1 FROM classify_api_logs WHERE id = %s AND api_key = %s",
            (log_id, api_key),
        )
        if cursor.fetchone() is None:
            cursor.close()
            return False
        cursor.execute(
            "UPDATE classify_api_logs SET prediction_label = %s, final_confidence = %s WHERE id = %s",
            (label, final_confidence, log_id),
        )
        conn.commit()
        cursor.close()
        return True
    finally:
        conn.close()
