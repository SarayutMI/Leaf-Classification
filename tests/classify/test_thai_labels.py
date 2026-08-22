# tests/classify/test_thai_labels.py
"""Thai names for the predicted classes.

Source: สรุประบบการคัดแยกพันธุ์ไม้.pdf pages 3-4, where the boxed entries in
each botanical chart are the classes these models support.
"""
import pytest

from src.classify.service import THAI_LABELS, mapping_predict_thai_name
from src.config import settings

HEADS = {
    "shape":  settings.SHAPE_CLASSES,
    "apex":   settings.APEX_CLASSES,
    "base":   settings.BASE_CLASSES,
    "margin": settings.MARGIN_CLASSES,
}


@pytest.mark.parametrize("region,classes", HEADS.items())
def test_every_configured_class_has_a_thai_name(region, classes):
    missing = [c for c in classes if c not in THAI_LABELS[region]]
    assert not missing, f"{region} classes without a Thai name: {missing}"


@pytest.mark.parametrize("region,classes", HEADS.items())
def test_no_stray_entries_beyond_the_configured_classes(region, classes):
    """A name for a class the model cannot emit means the two lists drifted."""
    extra = [k for k in THAI_LABELS[region] if k not in classes]
    assert not extra, f"{region} has Thai names for unsupported classes: {extra}"


def test_same_term_differs_by_region():
    """Caudate is a tail at the apex and a heart at the base — the whole reason
    the function takes a region."""
    assert mapping_predict_thai_name("apex", "Caudate") == "ยาวคล้ายหาง"
    assert mapping_predict_thai_name("base", "Caudate") == "รูปหัวใจ"


def test_known_labels():
    assert mapping_predict_thai_name("shape", "Cordate") == "รูปหัวใจ"
    assert mapping_predict_thai_name("apex", "Acute") == "แหลม"
    assert mapping_predict_thai_name("base", "Cuneate") == "รูปลิ่ม"
    assert mapping_predict_thai_name("margin", "Entire") == "เรียบ"


def test_region_and_label_are_case_insensitive():
    assert mapping_predict_thai_name("APEX", "obtuse") == "มน"
    assert mapping_predict_thai_name(" Margin ", "CRENATE") == "หยักมน"


@pytest.mark.parametrize("region,label", [
    ("apex", "Acuminate"),   # real botanical term, but not a supported class
    ("stem", "Acute"),       # region that has no table
])
def test_unknown_input_falls_back_to_english(region, label):
    """A missing translation must never turn a good prediction into a 500."""
    assert mapping_predict_thai_name(region, label) == label


def test_empty_label_passes_through():
    assert mapping_predict_thai_name("apex", "") == ""
    assert mapping_predict_thai_name("apex", None) is None
