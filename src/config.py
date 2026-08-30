# src/config.py
from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from the environment / .env.

    Values are validated and coerced on startup, so a malformed PORT or
    DB_PORT fails immediately instead of at first use.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Server ───────────────────────────────────────────────────
    PORT: int = 8000

    # ── Database ──────────────────────────────────────────────────
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_USER: str = "root"
    DB_PASSWORD: str = ""
    DB_NAME: str = "leaf_db"

    # ── AWS / S3 ──────────────────────────────────────────────────
    S3_BUCKET: str = ""
    AWS_DEFAULT_REGION: str = "ap-southeast-1"
    CDN_BASE: str = "https://d3w0s75zdffeft.cloudfront.net"
    S3_DATASET_PREFIX: str = "datasets"

    # Set AWS_ALLOWED_UPLOADED=false to skip uploading classified regions to S3.
    # AWS_ALLWED_UPLOADED (misspelled) is still read for existing deployments.
    AWS_ALLOWED_UPLOADED: bool = Field(
        default=True,
        validation_alias=AliasChoices("AWS_ALLOWED_UPLOADED", "AWS_ALLWED_UPLOADED"),
    )

    # Where the regions are written when AWS_ALLOWED_UPLOADED=false. The layout
    # under it mirrors the S3 keys (<S3_DATASET_PREFIX>/<region>/<file>.jpg), so
    # a local run can be synced to the bucket later as-is.
    LOCAL_UPLOAD_DIR: str = "./uploads"

    # ── Seed user ─────────────────────────────────────────────────
    SEED_USERNAME: str = "admin"
    SEED_PASSWORD: str | None = None

    # ── Model paths ───────────────────────────────────────────────
    # Accepts a .pt checkpoint or an exported OpenVINO folder
    # (see scripts/export_openvino.py) — ultralytics detects the format.
    YOLO_MODEL_PATH: str = "./Model-Leaf/yolo11s_leaf.pt"
    SHAPE_MODEL_PATH: str = "./Model_Classification/R50_Shape_final_V0.keras"
    APEX_MODEL_PATH: str = "./Model_Classification/R50_Apex_final_V1.keras"
    BASE_MODEL_PATH: str = "./Model_Classification/R50_Base_final_V1.keras"
    MARGIN_MODEL_PATH: str = "./Model_Classification/R50_Margin_final_V1.keras"

    # ── Class labels ─────────────────────────────────────────────
    SHAPE_CLASSES: list[str] = ["Cordate", "Lanceolate", "Ovate", "Sagittate"]
    APEX_CLASSES: list[str] = ["Acute", "Caudate", "Cuspidate", "Obtuse"]
    BASE_CLASSES: list[str] = ["Auriculate", "Caudate", "Cuneate", "Obtuse"]
    MARGIN_CLASSES: list[str] = ["Crenate", "Entire"]

    # ── Detection ─────────────────────────────────────────────────
    TARGET_CLASS: str = "leaf"
    # Longest side YOLO letterboxes the input to. Detection cost scales with
    # this, not with the uploaded resolution. Must match the imgsz baked into
    # the OpenVINO export (see scripts/export_openvino.py).
    YOLO_IMGSZ: int = 640
    # Landscape crop, kept on measurement rather than intuition. Portrait (3:4)
    # frames an upright leaf more tightly, but scored 96.7% against 4:3's 98.3%
    # on the 241 labelled photos (docs/experiments, scripts/compare_crop_aspect.py).
    # Raw bbox — what the dataset generator produces — came last at 94.6%, so
    # expanding the box helps the shape head whichever ratio is used.
    ASPECT_W: int = 4
    ASPECT_H: int = 3
    IMG_SIZE: int = 256

    # ── Load shedding ─────────────────────────────────────────────
    # Classification is serialized (one YOLO instance, non-thread-safe TFLite
    # interpreters), so extra concurrent callers only queue. Past this many
    # in flight the API returns 429 instead of letting clients time out.
    MAX_CONCURRENT_CLASSIFY: int = 4
    # Seconds a successful /health database ping stays cached, so a 30s
    # container healthcheck does not open a cross-network connection each time.
    HEALTH_DB_CACHE_SECONDS: float = 10.0

    # ── CPU tuning ────────────────────────────────────────────────
    # Match the deploy target (VPS: 2 cores). Used for TF thread pools
    # and TFLite interpreters.
    NUM_THREADS: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
