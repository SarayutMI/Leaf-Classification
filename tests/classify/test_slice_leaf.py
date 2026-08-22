# tests/classify/test_slice_leaf.py
"""The band bounds R50_Margin and R50_Base were trained on.

These numbers are not arbitrary: main:app.py sliced 0-0.30 / 0.30-0.60 /
0.70-1.0, and the classifiers were built against those crops. A refactor on
develop closed the 0.60-0.70 gap, feeding both heads 5% of the leaf they had
never been trained on. Pin the bounds so that cannot happen silently again.
"""
import numpy as np

from src.classify.service import slice_leaf


def test_bands_match_the_bounds_the_models_were_trained_on():
    crop = np.zeros((100, 40, 3), dtype=np.uint8)

    regions = slice_leaf(crop)

    assert regions["top"].shape[0] == 30       # 0.00 - 0.30
    assert regions["middle"].shape[0] == 30    # 0.30 - 0.60
    assert regions["bottom"].shape[0] == 30    # 0.70 - 1.00
    assert regions["full"].shape[0] == 100


def test_middle_and_bottom_do_not_touch():
    """The 0.60-0.70 transition zone belongs to neither head."""
    crop = np.zeros((100, 40, 3), dtype=np.uint8)
    # Row-numbered image, so each band reports which rows it received.
    for row in range(100):
        crop[row, :, :] = row % 256

    regions = slice_leaf(crop)
    middle_last = int(regions["middle"][-1, 0, 0])
    bottom_first = int(regions["bottom"][0, 0, 0])

    assert middle_last == 59
    assert bottom_first == 70


def test_full_is_the_whole_crop_untouched():
    crop = np.random.default_rng(0).integers(0, 255, (90, 120, 3), dtype=np.uint8)
    assert np.array_equal(slice_leaf(crop)["full"], crop)
