# services/s3.py
import uuid
import logging
import boto3
import cv2
import numpy as np
import config

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("s3", region_name=config.AWS_DEFAULT_REGION)
    return _client


def upload_regions(regions: dict, original_filename: str) -> dict[str, str]:
    """Upload 4 leaf regions to S3 and return {region: cdn_url}."""
    uid = uuid.uuid4().hex
    stem = original_filename.rsplit(".", 1)[0] if "." in original_filename else original_filename
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    filename = f"{safe_stem}_{uid}.jpg"

    client = _get_client()
    urls = {}

    for region_name, img in regions.items():
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok or buf is None:
            logger.warning("Failed to encode region '%s' (shape=%s), skipping", region_name, img.shape)
            continue
        key = f"{config.S3_DATASET_PREFIX}/{region_name}/{filename}"
        client.put_object(
            Bucket=config.S3_BUCKET,
            Key=key,
            Body=buf.tobytes(),
            ContentType="image/jpeg",
        )
        urls[region_name] = f"{config.CDN_BASE}/{key}"
        logger.info("Uploaded %s → %s", region_name, urls[region_name])

    return urls
