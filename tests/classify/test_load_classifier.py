# tests/classify/test_load_classifier.py
import pytest

from src.classify import service


@pytest.mark.parametrize("name", ["NewShape_.tflite", "NewShape_.keras"])
def test_missing_model_names_the_file(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match=f"Classifier model not found: {name}"):
        service._load_classifier(name)
