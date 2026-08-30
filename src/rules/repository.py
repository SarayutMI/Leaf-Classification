# src/rules/repository.py
"""Read-only access to the rule base.

One row = one exact combination of the four predicted traits, mapped to a
group name. The combination is UNIQUE, so a prediction can never match two
different groups exactly. The rows are written by the Laravel admin app; this
service only reads them while classifying.
"""
from src.database import get_conn

TRAIT_KEYS = ("shape", "apex", "base", "margin")

_COLUMNS = "id, code, name, shape, apex, base, margin, is_active"


def _row_to_group(row: dict) -> dict:
    row["is_active"] = bool(row["is_active"])
    return row


def list_groups(active_only: bool = False) -> list[dict]:
    conn = get_conn()
    try:
        cursor = conn.cursor(dictionary=True)
        sql = f"SELECT {_COLUMNS} FROM classify_rule_group"
        if active_only:
            sql += " WHERE is_active = 1"
        sql += " ORDER BY code"
        cursor.execute(sql)
        groups = [_row_to_group(row) for row in cursor.fetchall()]
        cursor.close()
        return groups
    finally:
        conn.close()
