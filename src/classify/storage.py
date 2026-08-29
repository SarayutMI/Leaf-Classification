# src/classify/storage.py
import uuid
import logging
from pathlib import Path

import boto3
import cv2
import numpy as np
from src.config import settings

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("s3", region_name=settings.AWS_DEFAULT_REGION)
    return _client


def new_dataset_filename(original_filename: str) -> str:
    """A collision-free, path-safe name for one upload's region set.

    Every non-alphanumeric character is replaced, so an original name shaped
    like a traversal ("../../etc/passwd.jpg") cannot escape the destination
    directory — this is what keeps save_regions_local() inside LOCAL_UPLOAD_DIR.

    The caller generates this before the upload runs, so the URLs the regions
    will land on can be returned to the client in the same response that
    triggered them (see region_urls).
    """
    stem = original_filename.rsplit(".", 1)[0] if "." in original_filename else original_filename
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return f"{safe_stem}_{uuid.uuid4().hex}.jpg"


def _region_key(region_name: str, dataset_filename: str) -> str:
    """The one place the <prefix>/<region>/<file> layout is spelled out.

    The S3 key, the CDN URL and the local path all derive from this, so the URL
    handed to the client cannot drift from the object actually written.
    """
    return f"{settings.S3_DATASET_PREFIX}/{region_name}/{dataset_filename}"


def region_urls(dataset_filename: str, region_names) -> dict[str, str]:
    """Where each region will be reachable once uploaded.

    Computed before the upload, not after it: the response is sent while
    upload_regions() is still running in a background task.
    """
    return {name: f"{settings.CDN_BASE}/{_region_key(name, dataset_filename)}"
            for name in region_names}


def _encode_jpeg(region_name: str, img: np.ndarray) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok or buf is None:
        logger.warning("Failed to encode region '%s' (shape=%s), skipping", region_name, img.shape)
        return None
    return buf.tobytes()


def upload_regions(regions: dict, dataset_filename: str) -> dict[str, str]:
    """Upload 4 leaf regions to S3 and return {region: cdn_url}.

    `dataset_filename` comes from new_dataset_filename() at the top of the
    request; a region whose JPEG encoding failed is absent from the result, so
    what is returned is what actually landed in the bucket.
    """
    client = _get_client()
    planned = region_urls(dataset_filename, regions)
    urls = {}

    for region_name, img in regions.items():
        body = _encode_jpeg(region_name, img)
        if body is None:
            continue
        client.put_object(
            Bucket=settings.S3_BUCKET,
            Key=_region_key(region_name, dataset_filename),
            Body=body,
            ContentType="image/jpeg",
        )
        urls[region_name] = planned[region_name]
        logger.info("Uploaded %s → %s", region_name, urls[region_name])

    return urls


def save_regions_local(regions: dict, dataset_filename: str) -> dict[str, str]:
    """Write the 4 leaf regions under LOCAL_UPLOAD_DIR, returning {region: path}.

    Used when AWS_ALLOWED_UPLOADED=false. The layout mirrors the S3 key layout
    (<prefix>/<region>/<file>.jpg) so a local run can be synced to the bucket
    later without rewriting anything. The returned paths go into the same
    classify_image_dataset.cdn_url column the CDN URLs use — but they are
    container-local filenames, not addresses, which is why the route does not
    hand them back to the caller.
    """
    root = Path(settings.LOCAL_UPLOAD_DIR)
    paths = {}

    for region_name, img in regions.items():
        body = _encode_jpeg(region_name, img)
        if body is None:
            continue
        dest = root / _region_key(region_name, dataset_filename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        paths[region_name] = str(dest)
        logger.info("Saved %s → %s", region_name, dest)

    return paths
