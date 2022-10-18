# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from splits.config import WorkerConfig

# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.datasets", "tests.fixtures.files", "tests.fixtures.hub"]


@pytest.fixture(scope="session")
def worker_config():
    worker_config = WorkerConfig()
    if "test" not in worker_config.cache.mongo_database or "test" not in worker_config.queue.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return worker_config
