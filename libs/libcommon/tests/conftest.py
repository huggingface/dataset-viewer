# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pytest import MonkeyPatch, fixture

from libcommon.config import CacheConfig, CommonConfig, QueueConfig, WorkerLoopConfig
from libcommon.processing_graph import ProcessingStep


@fixture(scope="session")
def common_config():
    return CommonConfig()


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@fixture(scope="session")
def monkeypatch_session():
    monkeypatch_session = MonkeyPatch()
    monkeypatch_session.setenv("CACHE_MONGO_DATABASE", "datasets_server_cache_test")
    monkeypatch_session.setenv("QUEUE_MONGO_DATABASE", "datasets_server_queue_test")
    yield monkeypatch_session
    monkeypatch_session.undo()


@fixture(scope="session", autouse=True)
def cache_config(monkeypatch_session: MonkeyPatch) -> CacheConfig:
    cache_config = CacheConfig()
    if "test" not in cache_config.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return cache_config


@fixture(scope="session", autouse=True)
def queue_config(monkeypatch_session: MonkeyPatch) -> QueueConfig:
    queue_config = QueueConfig()
    if "test" not in queue_config.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return queue_config


@fixture(scope="session", autouse=True)
def worker_loop_config(monkeypatch_session: MonkeyPatch) -> WorkerLoopConfig:
    return WorkerLoopConfig()


@fixture(scope="session")
def test_processing_step(monkeypatch_session: MonkeyPatch) -> ProcessingStep:
    return ProcessingStep(
        endpoint="/dummy",
        input_type="dataset",
        requires=None,
        required_by_dataset_viewer=False,
        parent=None,
        ancestors=[],
        children=[],
    )
