# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from libcache.config import CacheConfig


@pytest.fixture(scope="session")
def cache_config():
    config = CacheConfig()
    if "test" not in config.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return config
