"""The classifier heads were trained on directory-sorted classes.

Keras assigns output index i to sorted(class_dirs)[i], so any list in
config that is not sorted silently mislabels every prediction — the API
still answers 200 with a confident, wrong class. SHAPE_CLASSES shipped
permuted and scored 2/48 on the labelled crops in "Image for Example"
(46/48 once sorted), so pin the invariant.
"""
import pytest
from src.config import settings

HEADS = {
    "SHAPE_CLASSES":  settings.SHAPE_CLASSES,
    "APEX_CLASSES":   settings.APEX_CLASSES,
    "BASE_CLASSES":   settings.BASE_CLASSES,
    "MARGIN_CLASSES": settings.MARGIN_CLASSES,
}


@pytest.mark.parametrize("name", sorted(HEADS))
def test_class_list_is_in_training_order(name):
    classes = HEADS[name]
    assert classes == sorted(classes), (
        f"{name} must stay in the training (alphabetical) class order; "
        f"got {classes}, expected {sorted(classes)}"
    )


def test_shape_classes_match_reference_script():
    # Test-Model.py, the script used to evaluate R50_Shape against the
    # labelled test set, hardcodes this list.
    assert settings.SHAPE_CLASSES == ["Cordate", "Lanceolate", "Ovate", "Sagittate"]
