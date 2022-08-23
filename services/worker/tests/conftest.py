import os
from pathlib import Path

import pytest

from .utils import HF_ENDPOINT

# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.datasets", "tests.fixtures.files", "tests.fixtures.hub"]


@pytest.fixture(scope="session")
def config():
    return {"image_file": str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg")}


os.environ["HF_ENDPOINT"] = HF_ENDPOINT
