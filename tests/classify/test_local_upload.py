# tests/classify/test_local_upload.py
"""Where the region crops go when S3 is switched off.

AWS_ALLOWED_UPLOADED=false used to mean "discard the crops": _upload_and_save
logged and returned, so every deployment without bucket credentials threw away
its dataset. It now writes them under LOCAL_UPLOAD_DIR and records the paths in
the same classify_image_dataset row the CDN URLs use.
"""
from unittest.mock import patch

import numpy as np
import pytest

from src.classify import storage
from src.classify.router import _upload_and_save

# This module is the one that exercises storage for real.
pytestmark = pytest.mark.real_storage


@pytest.fixture
def regions():
    rng = np.random.default_rng(0)
    return {
        "full":   rng.integers(0, 255, (40, 30, 3), dtype=np.uint8),
        "top":    rng.integers(0, 255, (12, 30, 3), dtype=np.uint8),
        "middle": rng.integers(0, 255, (12, 30, 3), dtype=np.uint8),
        "bottom": rng.integers(0, 255, (12, 30, 3), dtype=np.uint8),
    }


@pytest.fixture
def local_dir(tmp_path):
    """Point storage at tmp_path. settings is a module-level singleton
    (@lru_cache in src/config.py), so patch the attribute, not the env."""
    with patch.object(storage.settings, "LOCAL_UPLOAD_DIR", str(tmp_path)):
        yield tmp_path


def test_upload_disabled_writes_every_region_to_the_local_dir(regions, local_dir):
    with patch("src.classify.router.settings.AWS_ALLOWED_UPLOADED", False), \
         patch("src.classify.router.repository") as mock_db:
        _upload_and_save(regions, "photo.jpg", log_id=7)

    written = sorted(p.relative_to(local_dir).parts[1] for p in local_dir.rglob("*.jpg"))
    assert written == ["bottom", "full", "middle", "top"]

    log_id, locations = mock_db.save_image_dataset.call_args.args
    assert log_id == 7
    assert set(locations) == {"full", "top", "middle", "bottom"}
    for path in locations.values():
        assert path.startswith(str(local_dir))


def test_saved_files_are_readable_jpegs(regions, local_dir):
    import cv2

    with patch("src.classify.router.settings.AWS_ALLOWED_UPLOADED", False), \
         patch("src.classify.router.repository"):
        _upload_and_save(regions, "photo.jpg", log_id=7)

    for path in local_dir.rglob("*.jpg"):
        assert cv2.imread(str(path)) is not None


def test_upload_enabled_uses_s3_and_writes_nothing_locally(regions, local_dir):
    with patch("src.classify.router.settings.AWS_ALLOWED_UPLOADED", True), \
         patch("src.classify.router.repository") as mock_db, \
         patch("src.classify.router.storage.upload_regions") as mock_upload:
        mock_upload.return_value = {"full": "https://cdn/full.jpg"}
        _upload_and_save(regions, "photo.jpg", log_id=7)

    mock_upload.assert_called_once()
    assert list(local_dir.rglob("*.jpg")) == []
    mock_db.save_image_dataset.assert_called_once_with(7, {"full": "https://cdn/full.jpg"})


def test_invalid_log_id_persists_nothing(regions, local_dir):
    with patch("src.classify.router.settings.AWS_ALLOWED_UPLOADED", False), \
         patch("src.classify.router.repository") as mock_db:
        _upload_and_save(regions, "photo.jpg", log_id=-1)

    assert list(local_dir.rglob("*.jpg")) == []
    mock_db.save_image_dataset.assert_not_called()


def test_traversal_shaped_filename_cannot_escape_the_upload_dir(regions, local_dir):
    """The uploaded filename reaches the path unmodified apart from
    sanitising — a name shaped like ../../ must not walk out of tmp_path."""
    with patch("src.classify.router.settings.AWS_ALLOWED_UPLOADED", False), \
         patch("src.classify.router.repository") as mock_db:
        _upload_and_save(regions, "../../etc/passwd.jpg", log_id=7)

    _, locations = mock_db.save_image_dataset.call_args.args
    for path in locations.values():
        assert local_dir in __import__("pathlib").Path(path).resolve().parents


def test_a_storage_failure_does_not_escape_the_background_task(regions, local_dir):
    """The task runs after the response is sent — an exception here must be
    logged, not raised into the event loop."""
    with patch("src.classify.router.settings.AWS_ALLOWED_UPLOADED", False), \
         patch("src.classify.router.repository"), \
         patch("src.classify.router.storage.save_regions_local", side_effect=OSError("disk full")):
        _upload_and_save(regions, "photo.jpg", log_id=7)
