import os

import pytest

from ._utils import HF_ENDPOINT


@pytest.fixture(scope="session")
def config():
    return {"image_file": os.path.join(os.path.dirname(__file__), "data", "test_image_rgb.jpg")}


os.environ["HF_ENDPOINT"] = HF_ENDPOINT
