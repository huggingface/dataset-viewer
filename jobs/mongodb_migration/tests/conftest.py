# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

# from typing import Iterator

from environs import Env

# from pytest import MonkeyPatch, fixture
from pytest import fixture

# from mongodb_migration.config import JobConfig
# from mongodb_migration.database_migrations import _clean_maintenance_database
# from mongodb_migration.resource import MigrationsDatabaseResource

# # see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
# @fixture(scope="session")
# def monkeypatch_session():
#     monkeypatch_session = MonkeyPatch()
#     monkeypatch_session.setenv("CACHE_MONGO_DATABASE", "datasets_server_cache_test")
#     monkeypatch_session.setenv("QUEUE_MONGO_DATABASE", "datasets_server_queue_test")
#     monkeypatch_session.setenv("MONGODB_MIGRATION_MONGO_DATABASE", "datasets_server_maintenance_test")
#     yield monkeypatch_session
#     monkeypatch_session.undo()


# @fixture(scope="session", autouse=True)
# def job_config(monkeypatch_session: MonkeyPatch) -> JobConfig:
#     job_config = JobConfig.from_env()
#     if (
#         "test" not in job_config.cache.mongo_database
#         or "test" not in job_config.queue.mongo_database
#         or "test" not in job_config.database_migrations.mongo_database
#     ):
#         raise ValueError("Test must be launched on a test mongo database")
#     return job_config


@fixture(scope="session")
def env() -> Env:
    return Env(expand_vars=True)


@fixture(scope="session")
def mongo_host(env: Env) -> str:
    try:
        return env.str(name="MONGODB_MIGRATION_MONGO_URL")
    except Exception as e:
        raise ValueError("MONGODB_MIGRATION_MONGO_URL is not set") from e
