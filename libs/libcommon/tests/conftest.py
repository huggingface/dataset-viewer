# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from environs import Env
from pytest import fixture

from libcommon.config import CacheConfig, QueueConfig

# CACHE_MONGO_URL and QUEUE_MONGO_URL must be set in the environment, and correspond to a running mongo database
env = Env(expand_vars=True)


@fixture()
def cache_config() -> CacheConfig:
    cache_config = CacheConfig(mongo_database="datasets_server_cache_test", mongo_url=env.str("CACHE_MONGO_URL"))
    if "test" not in cache_config.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return cache_config


@fixture()
def queue_config() -> QueueConfig:
    queue_config = QueueConfig(mongo_database="datasets_server_queue_test", mongo_url=env.str("QUEUE_MONGO_URL"))
    if "test" not in queue_config.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return queue_config
