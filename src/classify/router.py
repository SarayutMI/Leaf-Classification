# src/classify/router.py
import asyncio
import functools
import logging
import time
import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, UploadFile, Depends, Request
from fastapi.responses import JSONResponse, RedirectResponse
from src.auth.dependencies import require_api_key
from src.classify import repository, service, storage
from src.config import settings

router = APIRouter(tags=["classify"])
logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


def _log(api_key, ip, filename, http_status, status,
         error_message=None, shape=None, apex=None,
         base=None, margin=None, confidence=None, duration=None,
         prediction_label=None) -> int:
    try:
        return repository.log_api_call(
            api_key=api_key,
            ip_address=ip,
            filename=filename,
            http_status=http_status,
            status=status,
            error_message=error_message,
            shape=shape,
            apex=apex,
            base=base,
            margin=margin,
            confidence=confidence,
            duration=duration,
            prediction_label=prediction_label,
        )
    except Exception:
        logger.exception("Failed to write API log")
        return -1


def _upload_and_save(regions: dict, filename: str, log_id: int) -> None:
    if not settings.AWS_ALLOWED_UPLOADED:
        logger.info("S3 upload disabled (AWS_ALLOWED_UPLOADED=false) — skipping log_id=%s", log_id)
        return
    if log_id <= 0:
        logger.warning("Skipping S3 upload — no valid log_id (%s)", log_id)
        return
    try:
        cdn_urls = storage.upload_regions(regions, filename)
        repository.save_image_dataset(log_id, cdn_urls)
    except Exception:
        logger.exception("Background S3 upload failed for log_id=%s", log_id)


@router.post("/classify", include_in_schema=False)
async def classify_legacy():
    return RedirectResponse(url="/api/classify", status_code=308)


@router.post("/api/classify")
async def classify(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    api_key: str = Depends(require_api_key),
):
    ip = request.client.host if request.client else "unknown"
    filename = image.filename or "unknown"
    loop = asyncio.get_running_loop()
    started_at = time.perf_counter()

    try:
        file_bytes = await image.read()

        if len(file_bytes) > MAX_IMAGE_BYTES:
            background_tasks.add_task(_log, api_key, ip, filename, 400, "error", "Image too large")
            return JSONResponse(
                status_code=400,
                content={
                    "code": 400,
                    "status": "error",
                    "message": "Image too large",
                    "errors": {"type": "VALIDATION_ERROR", "details": f"Maximum image size is {MAX_IMAGE_BYTES // (1024 * 1024)} MB"},
                },
            )

        def _decode():
            arr = np.frombuffer(file_bytes, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        img = await loop.run_in_executor(None, _decode)
        if img is None:
            background_tasks.add_task(_log, api_key, ip, filename, 500, "error", "Unsupported image format")
            return JSONResponse(
                status_code=500,
                content={
                    "code": 500,
                    "status": "error",
                    "message": "Image invalid or cannot be processed",
                    "errors": {"type": "PROCESSING_ERROR", "details": "Unsupported image format"},
                },
            )

        cropped = await loop.run_in_executor(None, service.detect_leaf, img)
        if cropped is None:
            background_tasks.add_task(_log, api_key, ip, filename, 500, "error", "No leaf found")
            return JSONResponse(
                status_code=500,
                content={
                    "code": 500,
                    "status": "error",
                    "message": "Leaf not detected",
                    "errors": {"type": "PROCESSING_ERROR", "details": "No leaf found"},
                },
            )

        regions = service.slice_leaf(cropped)

        predictions = await loop.run_in_executor(None, service.predict_all, regions)
        shape_label,  shape_conf  = predictions["shape"]
        apex_label,   apex_conf   = predictions["apex"]
        base_label,   base_conf   = predictions["base"]
        margin_label, margin_conf = predictions["margin"]

        overall_conf = round((shape_conf + apex_conf + base_conf + margin_conf) / 4, 2)

        duration = round(time.perf_counter() - started_at, 3)

        log_id = await loop.run_in_executor(
            None,
            functools.partial(
                _log, api_key, ip, filename, 200, "success",
                shape=shape_label, apex=apex_label, base=base_label,
                margin=margin_label, confidence=overall_conf, duration=duration,
                prediction_label=None,
            ),
        )

        background_tasks.add_task(_upload_and_save, regions, filename, log_id)

        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "status": "success",
                "message": "Image processed successfully",
                "data": {
                    "shape":  shape_label,
                    "apex":   apex_label,
                    "base":   base_label,
                    "margin": margin_label,
                    "prediction": {"label": None, "confidence": overall_conf},
                },
            },
        )

    except Exception:
        logger.exception("Unhandled error in /classify")
        duration = round(time.perf_counter() - started_at, 3)
        background_tasks.add_task(_log, api_key, ip, filename, 500, "error", "An internal error occurred", duration=duration)
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "status": "error",
                "message": "An internal error occurred. Please try again.",
                "errors": {"type": "PROCESSING_ERROR", "details": "Contact support if this persists."},
            },
        )
