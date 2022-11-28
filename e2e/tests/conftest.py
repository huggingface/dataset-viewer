# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from .utils import poll

# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.files", "tests.fixtures.hub"]


@pytest.fixture(autouse=True, scope="session")
def ensure_services_are_up() -> None:
    assert poll("/", expected_code=404).status_code == 404
