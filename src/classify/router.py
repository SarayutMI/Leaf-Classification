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
# Bound directly rather than reached through `service`: it is a pure lookup with
# no model behind it, so tests that mock the service module still get real Thai
# names instead of a MagicMock the JSON encoder cannot serialize.
from src.classify.service import mapping_predict_thai_name
from src.config import settings
from src.rules import service as rules_service

router = APIRouter(tags=["classify"])
logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
RETRY_AFTER_SECONDS = 5

# Classifications currently being processed; see the 429 guard in classify().
_inflight = 0


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


def _upload_and_save(regions: dict, dataset_filename: str, log_id: int) -> None:
    """Persist the region crops and record where they landed.

    With AWS_ALLOWED_UPLOADED=false the regions go to LOCAL_UPLOAD_DIR instead
    of S3 — they used to be discarded, which silently cost every crop taken on
    a deployment without bucket credentials. Either way the locations are
    written to classify_image_dataset, so the caller cannot tell the difference.

    This runs AFTER the response has been sent, and that response already
    carries the CDN URLs these uploads are heading for (`data.images`). So the
    URLs are a promise, not a receipt: a client that fetches one immediately can
    beat the upload to it, and if the upload raises here, the URL stays dead
    while no classify_image_dataset row is written. A missing object is that
    failure surfacing, not a bug in the naming — the log line below is where it
    is recorded.
    """
    if log_id <= 0:
        logger.warning("Skipping region persistence — no valid log_id (%s)", log_id)
        return
    try:
        if settings.AWS_ALLOWED_UPLOADED:
            locations = storage.upload_regions(regions, dataset_filename)
        else:
            locations = storage.save_regions_local(regions, dataset_filename)
            logger.info(
                "S3 upload disabled (AWS_ALLOWED_UPLOADED=false) — saved %d region(s) "
                "under %s for log_id=%s", len(locations), settings.LOCAL_UPLOAD_DIR, log_id,
            )
        repository.save_image_dataset(log_id, locations)
    except Exception:
        logger.exception("Background region persistence failed for log_id=%s", log_id)


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

    # Shed load before doing any work. `_inflight` is only read and written
    # between awaits, so the event loop cannot interleave another request here.
    global _inflight
    if _inflight >= settings.MAX_CONCURRENT_CLASSIFY:
        background_tasks.add_task(_log, api_key, ip, filename, 429, "error", "Server busy")
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            content={
                "code": 429,
                "status": "error",
                "message": "Server busy — too many classifications in progress",
                "errors": {
                    "type": "RATE_LIMIT",
                    "details": f"At most {settings.MAX_CONCURRENT_CLASSIFY} images are processed at a time. Retry in {RETRY_AFTER_SECONDS}s.",
                },
            },
        )
    _inflight += 1

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

        # Name the region set now, not inside the background upload, so the
        # response can tell the caller where the crops are going. Local mode
        # produces container paths rather than addresses, so it reports nothing.
        dataset_filename = storage.new_dataset_filename(filename)
        image_urls = (storage.region_urls(dataset_filename, regions)
                      if settings.AWS_ALLOWED_UPLOADED else None)

        # Predict feaures with model
        predictions = await loop.run_in_executor(None, service.predict_all, regions)
        shape_label,  shape_conf  = predictions["shape"]
        apex_label,   apex_conf   = predictions["apex"]
        base_label,   base_conf   = predictions["base"]
        margin_label, margin_conf = predictions["margin"]
        
        overall_conf = round((shape_conf + apex_conf + base_conf + margin_conf) / 4, 2)

        # Classify with the decision tree. The rule base lives in MySQL and is
        # edited from /admin, so a bad rule or a DB blip must not turn a
        # successful classification into a 500 — degrade to no label instead.
        try:
            top_groups = rules_service.match_group(
                traits={"shape": shape_label, "apex": apex_label,
                        "base": base_label, "margin": margin_label},
                # predict_class returns percentages; match_group wants 0-1.
                probs={"shape": shape_conf / 100, "apex": apex_conf / 100,
                       "base": base_conf / 100, "margin": margin_conf / 100},
            )
            best = top_groups[0] if top_groups else None
        except Exception:
            logger.exception("Rule matching failed — returning no group label")
            best = None

        prediction_label = best["name"] if best else None

        duration = round(time.perf_counter() - started_at, 3)

        log_id = await loop.run_in_executor(
            None,
            functools.partial(
                _log, api_key, ip, filename, 200, "success",
                shape=shape_label, apex=apex_label, base=base_label,
                margin=margin_label, confidence=overall_conf, duration=duration,
                prediction_label=prediction_label,
            ),
        )

        background_tasks.add_task(_upload_and_save, regions, dataset_filename, log_id)

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
                    "shape_th":  mapping_predict_thai_name("shape",  shape_label),
                    "apex_th":   mapping_predict_thai_name("apex",   apex_label),
                    "base_th":   mapping_predict_thai_name("base",   base_label),
                    "margin_th": mapping_predict_thai_name("margin", margin_label),
                    # One CDN URL per segmentation, the same locations
                    # classify_image_dataset records. null when S3 is disabled.
                    "images": image_urls,
                    # group_id/code let the caller fetch the group's varieties.
                    "prediction": {
                        "group_id":   best["id"] if best else None,
                        "code":       best["code"] if best else None,
                        "label":      prediction_label,
                        "confidence": overall_conf,
                    },
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

    finally:
        _inflight -= 1
