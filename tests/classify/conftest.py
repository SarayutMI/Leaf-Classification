# tests/classify/conftest.py
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def stub_region_storage(request):
    """Keep route tests from persisting regions for real.

    TestClient runs background tasks, and _upload_and_save is one. Its
    AWS_ALLOWED_UPLOADED=false branch now writes JPEGs to disk instead of
    returning early, so a route test that mocks `service` would hand
    cv2.imencode a MagicMock and — depending on the .env in the checkout —
    either log an exception or scatter files into the working tree. Stub the
    storage module for every test in this package except the ones marked
    `real_storage`, which are testing that persistence itself.
    """
    if request.node.get_closest_marker("real_storage"):
        yield
        return
    with patch("src.classify.router.storage") as mock_storage:
        # The route puts storage.region_urls() straight into the JSON body, so
        # these two need real values — a MagicMock reaches the encoder and 500s
        # every route test in this package.
        mock_storage.new_dataset_filename.return_value = "photo_stub.jpg"
        mock_storage.region_urls.side_effect = lambda name, regions: {
            r: f"https://cdn.test/datasets/{r}/{name}" for r in regions
        }
        yield mock_storage
