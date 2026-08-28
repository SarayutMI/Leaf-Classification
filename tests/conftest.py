# tests/conftest.py
"""Shared pytest fixtures and import-path setup."""
import os
import sys

import pytest

# Make `src` importable when pytest is run from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_storage: let the test use the real src.classify.storage module "
        "instead of the stub in tests/classify/conftest.py",
    )


VALID_API_KEY = "abc123def456abc123def456abc123de"


@pytest.fixture
def valid_api_key() -> str:
    return VALID_API_KEY


@pytest.fixture
def jpeg_bytes() -> bytes:
    """A minimal valid JPEG, for endpoints that decode the upload."""
    import cv2
    import numpy as np

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()
