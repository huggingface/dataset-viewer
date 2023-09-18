# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Iterator

from libcommon.queue import _clean_queue_database
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import _clean_cache_database
from pytest import MonkeyPatch, fixture

from cache_maintenance.config import JobConfig


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@fixture(scope="session")
def monkeypatch_session() -> Iterator[MonkeyPatch]:
    monkeypatch_session = MonkeyPatch()
    monkeypatch_session.setenv("CACHE_MONGO_DATABASE", "datasets_server_cache_test")
    monkeypatch_session.setenv("QUEUE_MONGO_DATABASE", "datasets_server_queue_test")
    yield monkeypatch_session
    monkeypatch_session.undo()


@fixture(scope="session")
def job_config(monkeypatch_session: MonkeyPatch) -> JobConfig:
    job_config = JobConfig.from_env()
    if "test" not in job_config.cache.mongo_database or "test" not in job_config.queue.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return job_config


@fixture(autouse=True)
def cache_mongo_resource(job_config: JobConfig) -> Iterator[CacheMongoResource]:
    with CacheMongoResource(database=job_config.cache.mongo_database, host=job_config.cache.mongo_url) as resource:
        yield resource
        _clean_cache_database()


@fixture(autouse=True)
def queue_mongo_resource(job_config: JobConfig) -> Iterator[QueueMongoResource]:
    with QueueMongoResource(database=job_config.queue.mongo_database, host=job_config.queue.mongo_url) as resource:
        yield resource
        _clean_queue_database()
