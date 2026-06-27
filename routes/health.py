# routes/health.py
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from core import database
from services import leaf as leaf_svc

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health():
    db_status = "ok"
    db_error = None

    try:
        conn = database._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
        finally:
            conn.close()
    except Exception:
        db_status = "error"
        db_error = "Database unavailable"
        logger.exception("Health check DB ping failed")

    models_status = {
        "yolo":   "ok" if leaf_svc.yolo_model   is not None else "not_loaded",
        "shape":  "ok" if leaf_svc.shape_model  is not None else "not_loaded",
        "apex":   "ok" if leaf_svc.apex_model   is not None else "not_loaded",
        "base":   "ok" if leaf_svc.base_model   is not None else "not_loaded",
        "margin": "ok" if leaf_svc.margin_model is not None else "not_loaded",
    }
    models_ok = all(v == "ok" for v in models_status.values())

    ok = db_status == "ok" and models_ok

    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "code": 200 if ok else 503,
            "status": "ok" if ok else "error",
            "services": {
                "api":      "ok",
                "database": db_status,
                "models":   models_status,
            },
            **({"error": db_error} if db_error else {}),
        },
    )
