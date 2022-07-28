from os.path import dirname, join

import pytest


@pytest.fixture(scope="session")
def config():
    root = dirname(dirname(dirname(__file__)))
    return {"openapi": join(root, "chart", "static-files", "openapi.json")}
