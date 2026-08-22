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
    return _exists_in("classify_rule_group", "code = %s", (code,), exclude_id)


def combination_exists(data: dict, exclude_id: int | None = None) -> bool:
    """Another group already claims this exact trait combination."""
    return _exists_in(
        "classify_rule_group",
        "shape = %s AND apex = %s AND base = %s AND margin = %s",
        (data["shape"], data["apex"], data["base"], data["margin"]),
        exclude_id,
    )


def _exists_in(table: str, where: str, params: tuple, exclude_id: int | None) -> bool:
    conn = get_conn()
    try:
        cursor = conn.cursor()
        sql = f"SELECT id FROM {table} WHERE {where}"
        if exclude_id is not None:
            sql += " AND id <> %s"
            params = params + (exclude_id,)
        cursor.execute(sql, params)
        found = cursor.fetchone() is not None
        cursor.close()
        return found
    finally:
        conn.close()


# ── Varieties ────────────────────────────────────────────────
# A variety ("มันเสือ") belongs to exactly one rule group.

_VARIETY_COLUMNS = "id, group_id, name, is_active"


def _row_to_variety(row: dict) -> dict:
    row["is_active"] = bool(row["is_active"])
    return row


def list_varieties() -> list[dict]:
    conn = get_conn()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            f"SELECT {_VARIETY_COLUMNS} FROM classify_rule_variety "
            f"ORDER BY group_id, name"
        )
        varieties = [_row_to_variety(row) for row in cursor.fetchall()]
        cursor.close()
        return varieties
    finally:
        conn.close()


def get_variety(variety_id: int) -> dict | None:
    conn = get_conn()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            f"SELECT {_VARIETY_COLUMNS} FROM classify_rule_variety WHERE id = %s",
            (variety_id,),
        )
        row = cursor.fetchone()
        cursor.close()
        return _row_to_variety(row) if row else None
    finally:
        conn.close()


def create_varieties(group_id: int, items: list[dict]) -> list[int]:
    """Insert several varieties into one group, all or nothing."""
    conn = get_conn()
    try:
        cursor = conn.cursor()
        try:
            conn.start_transaction()
            ids = []
            for item in items:
                cursor.execute(
                    "INSERT INTO classify_rule_variety (group_id, name, is_active) "
                    "VALUES (%s, %s, %s)",
                    (group_id, item["name"], int(item["is_active"])),
                )
                ids.append(cursor.lastrowid)
            conn.commit()
            return ids
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
    finally:
        conn.close()


def get_varieties(variety_ids: list[int]) -> list[dict]:
    if not variety_ids:
        return []
    conn = get_conn()
    try:
        cursor = conn.cursor(dictionary=True)
        placeholders = ", ".join(["%s"] * len(variety_ids))
        cursor.execute(
            f"SELECT {_VARIETY_COLUMNS} FROM classify_rule_variety "
            f"WHERE id IN ({placeholders}) ORDER BY id",
            tuple(variety_ids),
        )
        varieties = [_row_to_variety(row) for row in cursor.fetchall()]
        cursor.close()
        return varieties
    finally:
        conn.close()


def update_variety(variety_id: int, data: dict) -> bool:
    """Update a variety. False if it no longer exists."""
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE classify_rule_variety SET group_id = %s, name = %s, "
            "is_active = %s WHERE id = %s",
            (data["group_id"], data["name"], int(data["is_active"]), variety_id),
        )
        conn.commit()
        # rowcount is 0 both for "row missing" and for "nothing changed".
        if cursor.rowcount == 0:
            cursor.execute(
                "SELECT id FROM classify_rule_variety WHERE id = %s", (variety_id,)
            )
            exists = cursor.fetchone() is not None
            cursor.close()
            return exists
        cursor.close()
        return True
    finally:
        conn.close()


def delete_variety(variety_id: int) -> bool:
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM classify_rule_variety WHERE id = %s", (variety_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        cursor.close()
        return deleted
    finally:
        conn.close()


def existing_variety_names(names: list[str], exclude_id: int | None = None) -> set[str]:
    """Which of these names are already taken, matched case-insensitively.

    One query for the whole batch, so a 20-name submission is still one round
    trip and every offending row can be reported at once.
    """
    if not names:
        return set()
    conn = get_conn()
    try:
        cursor = conn.cursor()
        placeholders = ", ".join(["%s"] * len(names))
        sql = (
            f"SELECT name FROM classify_rule_variety "
            f"WHERE LOWER(name) IN ({placeholders})"
        )
        params = tuple(n.lower() for n in names)
        if exclude_id is not None:
            sql += " AND id <> %s"
            params = params + (exclude_id,)
        cursor.execute(sql, params)
        found = {row[0].lower() for row in cursor.fetchall()}
        cursor.close()
        return found
    finally:
        conn.close()


def group_ids_that_exist(group_ids: list[int]) -> set[int]:
    if not group_ids:
        return set()
    conn = get_conn()
    try:
        cursor = conn.cursor()
        placeholders = ", ".join(["%s"] * len(group_ids))
        cursor.execute(
            f"SELECT id FROM classify_rule_group WHERE id IN ({placeholders})",
            tuple(group_ids),
        )
        found = {row[0] for row in cursor.fetchall()}
        cursor.close()
        return found
    finally:
        conn.close()


def variety_counts_by_group() -> dict[int, int]:
    """How many varieties each group has, so the rules table can show it."""
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT group_id, COUNT(*) FROM classify_rule_variety GROUP BY group_id"
        )
        counts = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()
        return counts
    finally:
        conn.close()
