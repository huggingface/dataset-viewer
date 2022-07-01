import os

import pytest


@pytest.fixture(scope="session")
def config():
    return {"image_file": os.path.join(os.path.dirname(__file__), "models", "data", "test_image_rgb.jpg")}
