# src/seed.py
import secrets

import mysql.connector

from src.auth.security import generate_api_key, hash_password
from src.config import settings

SEED_USERNAME = settings.SEED_USERNAME
SEED_PASSWORD = settings.SEED_PASSWORD or secrets.token_urlsafe(24)


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
