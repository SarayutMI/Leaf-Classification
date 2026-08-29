# tests/classify/test_image_urls.py
"""The CDN URLs /api/classify hands back for its four segmentations.

The regions were always uploaded and always recorded in classify_image_dataset;
what the caller never got was the address. The response now carries one URL per
region, and because it is sent before the background upload runs, the URL has to
be derived from a filename decided up front — which is why new_dataset_filename()
moved out of storage's upload functions and into the route.

The test that matters most is `test_the_response_urls_are_what_the_upload_records`:
the response and the database row must name the same object, and they are built
by two different code paths on opposite sides of the response.

Tests touching storage for real carry `real_storage`; the route tests deliberately
do not, so they get the stubbed storage from conftest.
"""
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.classify import storage
from src.classify.router import _upload_and_save, router

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)

VALID_KEY = "abc123def456abc123def456abc123de"
REGIONS = ("full", "top", "middle", "bottom")


@pytest.fixture
def regions():
    rng = np.random.default_rng(0)
    return {
        "full":   rng.integers(0, 255, (40, 30, 3), dtype=np.uint8),
        "top":    rng.integers(0, 255, (12, 30, 3), dtype=np.uint8),
        "middle": rng.integers(0, 255, (12, 30, 3), dtype=np.uint8),
        "bottom": rng.integers(0, 255, (12, 30, 3), dtype=np.uint8),
    }


# ── naming and URL layout ──────────────────────────────────────────────────

@pytest.mark.real_storage
def test_every_region_gets_a_url_under_its_own_prefix():
    urls = storage.region_urls("leaf_abc123.jpg", REGIONS)

    assert set(urls) == set(REGIONS)
    for region, url in urls.items():
        assert url == (f"{storage.settings.CDN_BASE}/"
                       f"{storage.settings.S3_DATASET_PREFIX}/{region}/leaf_abc123.jpg")


@pytest.mark.real_storage
def test_all_four_regions_share_one_filename():
    """One upload is one set of rows — a per-region name would break the join
    between a leaf's four crops."""
    urls = storage.region_urls(storage.new_dataset_filename("photo.jpg"), REGIONS)

    filenames = {url.rsplit("/", 1)[1] for url in urls.values()}
    assert len(filenames) == 1


@pytest.mark.real_storage
def test_each_request_gets_a_fresh_filename():
    assert storage.new_dataset_filename("photo.jpg") != storage.new_dataset_filename("photo.jpg")


@pytest.mark.real_storage
def test_the_response_urls_are_what_the_upload_records(regions):
    """The URL handed to the caller and the URL stored in the DB are the same
    string. region_urls() runs before the response and upload_regions() after
    it, so this is the seam where the two can drift apart."""
    dataset_filename = storage.new_dataset_filename("photo.jpg")
    promised = storage.region_urls(dataset_filename, regions)

    with patch("src.classify.router.settings.AWS_ALLOWED_UPLOADED", True), \
         patch("src.classify.router.repository") as mock_db, \
         patch("src.classify.storage._get_client"):
        _upload_and_save(regions, dataset_filename, log_id=7)

    log_id, recorded = mock_db.save_image_dataset.call_args.args
    assert log_id == 7
    assert recorded == promised


# ── the route ──────────────────────────────────────────────────────────────

def _classify(jpeg_bytes):
    with patch("src.classify.router.repository") as mock_repo, \
         patch("src.classify.router.service.detect_leaf"), \
         patch("src.classify.router.service.slice_leaf") as mock_slice, \
         patch("src.classify.router.service.predict_all") as mock_predict, \
         patch("src.classify.router.rules_service") as mock_rules, \
         patch("src.auth.dependencies.auth_repository") as mock_tokens:
        mock_tokens.get_token_by_api_key.return_value = {"api_key": VALID_KEY}
        mock_rules.match_group.return_value = []
        mock_slice.return_value = {r: object() for r in REGIONS}
        mock_predict.return_value = {
            "shape":  ("Cordate", 90.0),
            "apex":   ("Acute", 85.0),
            "base":   ("Cuneate", 80.0),
            "margin": ("Entire", 75.0),
        }
        mock_repo.log_api_call.return_value = 1

        return client.post(
            "/api/classify",
            headers={"X-API-Key": VALID_KEY},
            files={"image": ("leaf.jpg", jpeg_bytes, "image/jpeg")},
        )


def test_response_carries_one_image_url_per_segmentation(jpeg_bytes):
    # Pinned rather than inherited: the checkout's .env decides this flag, and
    # it is false in local development.
    with patch("src.classify.router.settings.AWS_ALLOWED_UPLOADED", True):
        response = _classify(jpeg_bytes)

    assert response.status_code == 200
    data = response.json()["data"]

    assert set(data["images"]) == set(REGIONS)
    assert all(isinstance(url, str) for url in data["images"].values())

    # The traits are untouched — existing clients must not break.
    assert data["shape"] == "Cordate"
    assert data["prediction"]["confidence"] == 82.5


def test_local_mode_reports_no_urls(jpeg_bytes):
    """With S3 off the crops land on container-local paths, which are not
    addresses. Reporting them as URLs would hand the caller a dead link."""
    with patch("src.classify.router.settings.AWS_ALLOWED_UPLOADED", False):
        response = _classify(jpeg_bytes)

    assert response.status_code == 200
    assert response.json()["data"]["images"] is None
