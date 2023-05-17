# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
from typing import Optional

import pytest
from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230309141600_cache_add_job_runner_version import (
    MigrationAddJobRunnerVersionToCacheResponse,
)


def test_cache_add_job_runner_version_without_worker_version(mongo_host: str) -> None:
    with MongoResource(database="test_cache_add_job_runner_version", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].insert_many(
            [{"kind": "/splits", "dataset": "dataset_without_worker_version", "http_status": 200}]
        )
        migration = MigrationAddJobRunnerVersionToCacheResponse(
            version="20230309141600", description="add 'job_runner_version' field based on 'worker_version' value"
        )
        migration.up()
        result = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": "dataset_without_worker_version"})
        assert result
        assert not result["job_runner_version"]
        db[CACHE_COLLECTION_RESPONSES].drop()


@pytest.mark.parametrize(
    "worker_version,expected",
    [
        ("2.0.0", 2),
        ("1.5.0", 1),
        ("WrongFormat", None),
        (None, None),
    ],
)
def test_cache_add_job_runner_version(mongo_host: str, worker_version: str, expected: Optional[int]) -> None:
    with MongoResource(database="test_cache_add_job_runner_version", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].insert_many(
            [{"kind": "/splits", "dataset": "dataset", "http_status": 200, "worker_version": worker_version}]
        )
        migration = MigrationAddJobRunnerVersionToCacheResponse(
            version="20230309141600", description="add 'job_runner_version' field based on 'worker_version' value"
        )
        migration.up()
        result = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": "dataset"})
        assert result
        assert result["job_runner_version"] == expected
        db[CACHE_COLLECTION_RESPONSES].drop()
