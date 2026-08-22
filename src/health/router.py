# src/health/router.py
import logging
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.classify import service as leaf_service
from src.config import settings
from src.database import get_conn

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)

# (checked_at, ok) of the last database ping. The container healthcheck runs
# every 30s and the database is across the network, so a successful result is
# reused for HEALTH_DB_CACHE_SECONDS. Failures are never cached — a recovering
# database must be reported as soon as it answers again.
_db_probe: tuple[float, bool] = (0.0, False)


def _ping_database() -> bool:
    global _db_probe

    checked_at, was_ok = _db_probe
    if was_ok and (time.monotonic() - checked_at) < settings.HEALTH_DB_CACHE_SECONDS:
        return True

    try:
        conn = get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
        finally:
            conn.close()
    except Exception:
        logger.exception("Health check DB ping failed")
        _db_probe = (time.monotonic(), False)
        return False

    _db_probe = (time.monotonic(), True)
    return True


def _models_status() -> dict[str, str]:
    return {
        "yolo":   "ok" if leaf_service.yolo_model   is not None else "not_loaded",
        "shape":  "ok" if leaf_service.shape_model  is not None else "not_loaded",
        "apex":   "ok" if leaf_service.apex_model   is not None else "not_loaded",
        "base":   "ok" if leaf_service.base_model   is not None else "not_loaded",
        "margin": "ok" if leaf_service.margin_model is not None else "not_loaded",
    }


@router.get("/health/live")
async def live():
    """Liveness only — the process is up and serving. Touches nothing external,
    so an uptime monitor can poll it as often as it likes."""
    return JSONResponse(
        status_code=200,
        content={"code": 200, "status": "ok", "services": {"api": "ok"}},
    )


@router.get("/health")
async def health():
    db_ok = _ping_database()
    models_status = _models_status()
    ok = db_ok and all(v == "ok" for v in models_status.values())

    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "code": 200 if ok else 503,
            "status": "ok" if ok else "error",
            "services": {
                "api":      "ok",
                "database": "ok" if db_ok else "error",
                "models":   models_status,
            },
            **({} if db_ok else {"error": "Database unavailable"}),
        },
    )
