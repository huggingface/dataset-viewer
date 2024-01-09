# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from http import HTTPStatus

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20240109160700_cache_add_failed_runs import MigrationAddFailedRunsToCacheResponse


def test_cache_add_failed_runs_to_jobs(mongo_host: str) -> None:
    with MongoResource(database="test_cache_add_failed_runs_to_jobs", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].insert_many(
            [
                {
                    "kind": "/splits",
                    "dataset": "success",
                    "http_status": HTTPStatus.OK,
                },
                {
                    "kind": "/splits",
                    "dataset": "error",
                    "http_status": HTTPStatus.INTERNAL_SERVER_ERROR,
                },
            ]
        )

        migration = MigrationAddFailedRunsToCacheResponse(
            version="20240109160700", description="add 'failed_runs' filed to cache records"
        )
        migration.up()

        result = list(db[CACHE_COLLECTION_RESPONSES].find({"dataset": "success"}))
        assert len(result) == 1
        assert result[0]["failed_runs"] == 0

        result = list(db[CACHE_COLLECTION_RESPONSES].find({"dataset": "error"}))
        assert len(result) == 1
        assert result[0]["failed_runs"] == 1

        migration.down()
        result = list(db[CACHE_COLLECTION_RESPONSES].find())
        assert "failed_runs" not in result[0]
        assert "failed_runs" not in result[1]

        db[CACHE_COLLECTION_RESPONSES].drop()
