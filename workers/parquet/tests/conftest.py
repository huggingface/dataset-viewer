# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pytest import MonkeyPatch, fixture

from parquet.config import WorkerConfig

# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.datasets", "tests.fixtures.files", "tests.fixtures.hub"]


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@fixture(scope="session")
def monkeypatch_session(hf_endpoint: str, hf_token: str):
    monkeypatch_session = MonkeyPatch()
    monkeypatch_session.setenv("CACHE_MONGO_DATABASE", "datasets_server_cache_test")
    monkeypatch_session.setenv("QUEUE_MONGO_DATABASE", "datasets_server_queue_test")
    monkeypatch_session.setenv("COMMON_HF_ENDPOINT", hf_endpoint)
    monkeypatch_session.setenv("COMMON_HF_TOKEN", hf_token)
    yield monkeypatch_session
    monkeypatch_session.undo()


@fixture(scope="session", autouse=True)
def worker_config(monkeypatch_session: MonkeyPatch) -> WorkerConfig:
    worker_config = WorkerConfig()
    if "test" not in worker_config.cache.mongo_database or "test" not in worker_config.queue.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return worker_config
