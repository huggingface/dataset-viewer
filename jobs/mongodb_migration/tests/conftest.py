# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pytest import MonkeyPatch, fixture

from mongodb_migration.config import JobConfig


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@fixture(scope="session")
def monkeypatch_session():
    monkeypatch_session = MonkeyPatch()
    monkeypatch_session.setenv("CACHE_MONGO_DATABASE", "datasets_server_cache_test")
    monkeypatch_session.setenv("QUEUE_MONGO_DATABASE", "datasets_server_queue_test")
    monkeypatch_session.setenv("MONGODB_MIGRATION_MONGO_DATABASE", "datasets_server_maintenance_test")
    yield monkeypatch_session
    monkeypatch_session.undo()


@fixture(scope="session", autouse=True)
def app_config(monkeypatch_session: MonkeyPatch) -> JobConfig:
    job_config = JobConfig()
    if (
        "test" not in job_config.cache.mongo_database
        or "test" not in job_config.queue.mongo_database
        or "test" not in job_config.mongodb_migration.mongo_database
    ):
        raise ValueError("Test must be launched on a test mongo database")
    return job_config
