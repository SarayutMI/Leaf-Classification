# src/rules/repository.py
"""Raw-SQL data access for the rule base.

One row = one exact combination of the four predicted traits, mapped to a
group name. The combination is UNIQUE, so a prediction can never match two
different groups exactly.
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


def get_group(group_id: int) -> dict | None:
    conn = get_conn()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            f"SELECT {_COLUMNS} FROM classify_rule_group WHERE id = %s",
            (group_id,),
        )
        row = cursor.fetchone()
        cursor.close()
        return _row_to_group(row) if row else None
    finally:
        conn.close()


def create_group(data: dict) -> int:
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO classify_rule_group "
            "(code, name, shape, apex, base, margin, is_active) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (data["code"], data["name"], data["shape"], data["apex"],
             data["base"], data["margin"], int(data["is_active"])),
        )
        conn.commit()
        group_id = cursor.lastrowid
        cursor.close()
        return group_id
    finally:
        conn.close()


def update_group(group_id: int, data: dict) -> bool:
    """Update a group. False if it no longer exists."""
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE classify_rule_group SET code = %s, name = %s, shape = %s, "
            "apex = %s, base = %s, margin = %s, is_active = %s WHERE id = %s",
            (data["code"], data["name"], data["shape"], data["apex"],
             data["base"], data["margin"], int(data["is_active"]), group_id),
        )
        conn.commit()
        # rowcount is 0 both for "row missing" and for "nothing changed", so
        # check existence rather than reporting a no-op edit as a 404.
        if cursor.rowcount == 0:
            cursor.execute(
                "SELECT id FROM classify_rule_group WHERE id = %s", (group_id,)
            )
            exists = cursor.fetchone() is not None
            cursor.close()
            return exists
        cursor.close()
        return True
    finally:
        conn.close()


def delete_group(group_id: int) -> bool:
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM classify_rule_group WHERE id = %s", (group_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        cursor.close()
        return deleted
    finally:
        conn.close()


def code_exists(code: str, exclude_id: int | None = None) -> bool:
    return _exists(
        "code = %s", (code,), exclude_id,
    )


def combination_exists(data: dict, exclude_id: int | None = None) -> bool:
    """Another group already claims this exact trait combination."""
    return _exists(
        "shape = %s AND apex = %s AND base = %s AND margin = %s",
        (data["shape"], data["apex"], data["base"], data["margin"]),
        exclude_id,
    )


def _exists(where: str, params: tuple, exclude_id: int | None) -> bool:
    conn = get_conn()
    try:
        cursor = conn.cursor()
        sql = f"SELECT id FROM classify_rule_group WHERE {where}"
        if exclude_id is not None:
            sql += " AND id <> %s"
            params = params + (exclude_id,)
        cursor.execute(sql, params)
        found = cursor.fetchone() is not None
        cursor.close()
        return found
    finally:
        conn.close()
