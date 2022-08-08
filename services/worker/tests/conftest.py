import os
from pathlib import Path

import pytest

from ._utils import HF_ENDPOINT


@pytest.fixture(scope="session")
def config():
    return {"image_file": str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg")}


os.environ["HF_ENDPOINT"] = HF_ENDPOINT
