# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from libqueue.config import QueueConfig


@pytest.fixture(scope="session")
def queue_config():
    config = QueueConfig()
    if "test" not in config.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return config
