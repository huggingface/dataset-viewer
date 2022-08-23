import os
from pathlib import Path

import pytest

from .utils import HF_ENDPOINT


@pytest.fixture(scope="session")
def config():
    return {"image_file": str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg")}


# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.datasets", "tests.fixtures.files", "tests.fixtures.hub"]


os.environ["HF_ENDPOINT"] = HF_ENDPOINT
