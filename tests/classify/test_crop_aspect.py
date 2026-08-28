# tests/classify/test_crop_aspect.py
"""The crop target the region bands are cut out of.

`slice_leaf()` cuts at fixed fractions of HEIGHT, so the aspect the bbox is
expanded to decides how much of each band is leaf and how much is background.
Nothing asserted this before: test_slice_leaf.py pins the band fractions but
passes under any aspect.

4:3 is a measured choice, not a default. Portrait 3:4 frames an upright leaf
more tightly and was tried; it scored 96.7% against 4:3's 98.3% over the 241
labelled photos, and the raw bounding box came last at 94.6%
(scripts/compare_crop_aspect.py). Pin it so a change is deliberate and gets
re-measured.
"""
import numpy as np
import pytest

from src.classify.service import _crop_to_aspect
from src.config import settings

TARGET = settings.ASPECT_W / settings.ASPECT_H


def _blank(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_settings_target_is_the_measured_ratio():
    assert (settings.ASPECT_W, settings.ASPECT_H) == (4, 3)


def test_portrait_bbox_is_expanded_to_the_target_ratio():
    # Taller than 4:3 — width is the side that grows.
    crop = _crop_to_aspect(_blank(2000, 2000), 900, 500, 1100, 1500)

    h, w = crop.shape[:2]
    assert w / h == pytest.approx(TARGET, abs=0.01)
    assert h == 1000            # the long side is untouched
    assert w >= 200             # never narrower than the original bbox


def test_landscape_bbox_is_expanded_to_the_target_ratio():
    # Wider than 4:3 — height is the side that grows.
    crop = _crop_to_aspect(_blank(2000, 2000), 500, 900, 1500, 1100)

    h, w = crop.shape[:2]
    assert w / h == pytest.approx(TARGET, abs=0.01)
    assert w == 1000            # the long side is untouched
    assert h >= 200


def test_crop_never_shrinks_below_the_bounding_box():
    image = _blank(2000, 2000)
    for box in [(900, 500, 1100, 1500), (500, 900, 1500, 1100), (0, 0, 400, 400)]:
        x1, y1, x2, y2 = box
        crop = _crop_to_aspect(image, x1, y1, x2, y2)
        assert crop.shape[0] >= y2 - y1
        assert crop.shape[1] >= x2 - x1


def test_bbox_against_the_image_edge_keeps_the_full_expanded_size():
    """The re-anchor path: a box clamped at the border slides inward
    instead of returning a crop at the wrong ratio."""
    crop = _crop_to_aspect(_blank(2000, 2000), 0, 0, 200, 1000)

    h, w = crop.shape[:2]
    assert w / h == pytest.approx(TARGET, abs=0.01)
    assert h == 1000


def test_crop_larger_than_the_image_is_clamped_not_padded():
    """A bbox filling a small image cannot be expanded — the result is
    clamped to the image, so the ratio is not reachable. It must still be a
    valid, non-empty crop inside the image."""
    crop = _crop_to_aspect(_blank(100, 100), 0, 0, 100, 100)

    assert crop.shape[0] == 100
    assert crop.shape[1] <= 100
    assert crop.size > 0
