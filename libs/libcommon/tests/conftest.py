# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from environs import Env
from pytest import fixture

from libcommon.config import CacheConfig, QueueConfig
from libcommon.processing_graph import ProcessingStep

# CACHE_MONGO_URL and QUEUE_MONGO_URL must be set in the environment, and correspond to a running mongo database
env = Env(expand_vars=True)


@fixture(scope="session", autouse=True)
def cache_config() -> CacheConfig:
    cache_config = CacheConfig(mongo_database="datasets_server_cache_test", mongo_url=env.str("CACHE_MONGO_URL"))
    if "test" not in cache_config.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return cache_config


@fixture(scope="session", autouse=True)
def queue_config() -> QueueConfig:
    queue_config = QueueConfig(mongo_database="datasets_server_queue_test", mongo_url=env.str("QUEUE_MONGO_URL"))
    if "test" not in queue_config.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return queue_config


@fixture(scope="session")
def test_processing_step() -> ProcessingStep:
    return ProcessingStep(
        endpoint="/dummy",
        input_type="dataset",
        requires=None,
        required_by_dataset_viewer=False,
        parent=None,
        ancestors=[],
        children=[],
    )
