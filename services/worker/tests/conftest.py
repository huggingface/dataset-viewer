import os

from .utils import HF_ENDPOINT

# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.datasets", "tests.fixtures.files", "tests.fixtures.hub"]


os.environ["HF_ENDPOINT"] = HF_ENDPOINT
