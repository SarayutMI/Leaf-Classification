# src/seed.py
import secrets

import mysql.connector

from src.auth.security import generate_api_key, hash_password
from src.config import settings

SEED_USERNAME = settings.SEED_USERNAME
SEED_PASSWORD = settings.SEED_PASSWORD or secrets.token_urlsafe(24)


def _ensure_column(cursor, table: str, column: str, ddl: str) -> None:
    """Add a column if it is missing. Idempotent — safe on every startup."""
    cursor.execute(
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s AND column_name = %s",
        (settings.DB_NAME, table, column),
    )
    if cursor.fetchone()[0] == 0:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _table_exists(cursor, table: str) -> bool:
    cursor.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_schema = %s AND table_name = %s",
        (settings.DB_NAME, table),
    )
    return cursor.fetchone()[0] > 0


def _ensure_index(cursor, table: str, index: str, ddl: str) -> None:
    """Add an index if it is missing. Idempotent — safe on every startup."""
    cursor.execute(
        "SELECT COUNT(*) FROM information_schema.statistics "
        "WHERE table_schema = %s AND table_name = %s AND index_name = %s",
        (settings.DB_NAME, table, index),
    )
    if cursor.fetchone()[0] == 0:
        cursor.execute(f"ALTER TABLE {table} ADD {ddl}")


def _migrate_rule_traits(cursor) -> None:
    """Fold the old classify_rule_trait child rows into trait columns.

    The rule base first shipped with one child row per allowed value. Every
    trait now holds exactly one class, so the values move onto the group row.
    Nothing is dropped: the old table and the unused ar_min/ar_max/priority
    columns are left in place for a manual cleanup once this is verified.
    """
    for key in ("shape", "apex", "base", "margin"):
        _ensure_column(cursor, "classify_rule_group", key, "VARCHAR(50) NOT NULL DEFAULT ''")

    if _table_exists(cursor, "classify_rule_trait"):
        for key in ("shape", "apex", "base", "margin"):
            cursor.execute(
                f"""
                UPDATE classify_rule_group g
                  JOIN (SELECT group_id, MIN(trait_value) AS value
                          FROM classify_rule_trait
                         WHERE trait_key = %s AND trait_value <> '*'
                      GROUP BY group_id) t ON t.group_id = g.id
                   SET g.{key} = t.value
                 WHERE g.{key} = ''
                """,
                (key,),
            )
            if cursor.rowcount:
                print(f"Migrated {cursor.rowcount} rule group(s): {key}")

    _ensure_index(
        cursor,
        "classify_rule_group",
        "uq_rule_combination",
        "UNIQUE KEY uq_rule_combination (shape, apex, base, margin)",
    )


def run():
    conn = mysql.connector.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
    )
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classify_user (
            id            INT AUTO_INCREMENT PRIMARY KEY,
            username      VARCHAR(100) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            created_at    DATETIME DEFAULT NOW()
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classify_token (
            id         INT AUTO_INCREMENT PRIMARY KEY,
            user_id    INT NOT NULL,
            api_key    VARCHAR(64) NOT NULL UNIQUE,
            created_at DATETIME DEFAULT NOW(),
            FOREIGN KEY (user_id) REFERENCES classify_user(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classify_api_logs (
            id            INT AUTO_INCREMENT PRIMARY KEY,
            api_key       VARCHAR(64)  NOT NULL,
            ip_address    VARCHAR(45)  NOT NULL,
            filename      VARCHAR(255) NOT NULL,
            http_status   INT          NOT NULL,
            status        VARCHAR(10)  NOT NULL,
            error_message TEXT         NULL,
            shape         VARCHAR(50)  NULL,
            apex          VARCHAR(50)  NULL,
            base          VARCHAR(50)  NULL,
            margin        VARCHAR(50)  NULL,
            confidence    FLOAT        NULL,
            duration      FLOAT        NULL,
            prediction_label VARCHAR(255) NULL,
            called_at     DATETIME     DEFAULT NOW()
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classify_image_dataset (
            id         INT AUTO_INCREMENT PRIMARY KEY,
            log_id     INT          NOT NULL,
            region     VARCHAR(10)  NOT NULL COMMENT 'full | top | middle | bottom',
            cdn_url    TEXT         NOT NULL,
            created_at DATETIME     DEFAULT NOW(),
            FOREIGN KEY (log_id) REFERENCES classify_api_logs(id)
        )
    """)

    # One rule = one exact combination of the four predicted traits mapped to a
    # group name. Every trait holds exactly one class, so there is nothing for a
    # child table to hold and the combination itself is the natural key.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classify_rule_group (
            id         INT AUTO_INCREMENT PRIMARY KEY,
            code       VARCHAR(20)  NOT NULL UNIQUE COMMENT 'G1, G2, ...',
            name       VARCHAR(255) NOT NULL COMMENT 'group name, e.g. กลุ่มใบหัวใจ',
            shape      VARCHAR(50)  NOT NULL,
            apex       VARCHAR(50)  NOT NULL,
            base       VARCHAR(50)  NOT NULL,
            margin     VARCHAR(50)  NOT NULL,
            is_active  TINYINT(1)   NOT NULL DEFAULT 1,
            created_at DATETIME     DEFAULT NOW(),
            updated_at DATETIME     DEFAULT NOW() ON UPDATE NOW(),
            UNIQUE KEY uq_rule_combination (shape, apex, base, margin)
        )
    """)

    _migrate_rule_traits(cursor)

    # Databases created before these columns existed still need them: the
    # CREATE TABLE above is a no-op once the table exists, and
    # classify/repository.log_api_call writes both on every call.
    _ensure_column(cursor, "classify_api_logs", "duration", "FLOAT NULL")
    _ensure_column(cursor, "classify_api_logs", "prediction_label", "VARCHAR(255) NULL")

    conn.commit()
    print("Tables created.")

    cursor.execute(
        "SELECT id FROM classify_user WHERE username = %s",
        (SEED_USERNAME,),
    )
    existing = cursor.fetchone()

    if existing:
        print(f"Seed user '{SEED_USERNAME}' already exists. Skipping.")
        cursor.execute(
            "SELECT api_key FROM classify_token WHERE user_id = %s",
            (existing[0],),
        )
        row = cursor.fetchone()
        if row:
            print(f"  api_key  : {row[0]}")
    else:
        password_hash = hash_password(SEED_PASSWORD)
        cursor.execute(
            "INSERT INTO classify_user (username, password_hash) VALUES (%s, %s)",
            (SEED_USERNAME, password_hash),
        )
        conn.commit()
        user_id = cursor.lastrowid

        api_key = generate_api_key()
        cursor.execute(
            "INSERT INTO classify_token (user_id, api_key) VALUES (%s, %s)",
            (user_id, api_key),
        )
        conn.commit()

        print("Seed user created. Save these credentials — they will not be shown again.")
        print(f"  username : {SEED_USERNAME}")
        print(f"  password : {SEED_PASSWORD}")
        print(f"  api_key  : {api_key}")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    run()
