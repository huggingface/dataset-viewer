# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from admin.config import AppConfig

# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.hub"]


@pytest.fixture(scope="session")
def app_config():
    app_config = AppConfig()
    if "test" not in app_config.cache.mongo_database or "test" not in app_config.queue.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return app_config
