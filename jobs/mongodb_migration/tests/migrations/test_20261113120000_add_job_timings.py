# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

from datetime import datetime

from libcommon.constants import (
    CACHE_COLLECTION_RESPONSES,
    CACHE_MONGOENGINE_ALIAS,
    QUEUE_COLLECTION_JOBS,
    QUEUE_MONGOENGINE_ALIAS,
)
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20261113120000_add_job_timings import CACHE_FIELDS, MigrationAddJobTimings


def test_add_job_timings(mongo_host: str) -> None:
    with (
        MongoResource(database="test_add_job_timings_cache", host=mongo_host, mongoengine_alias="cache"),
        MongoResource(database="test_add_job_timings_queue", host=mongo_host, mongoengine_alias="queue"),
    ):
        now = datetime(2026, 11, 13)
        cache = get_db(CACHE_MONGOENGINE_ALIAS)[CACHE_COLLECTION_RESPONSES]
        jobs = get_db(QUEUE_MONGOENGINE_ALIAS)[QUEUE_COLLECTION_JOBS]
        cache.insert_many([{"dataset": "old"}, {"dataset": "new", **dict.fromkeys(CACHE_FIELDS, now)}])
        jobs.insert_many([{"dataset": "old"}, {"dataset": "new", "first_job_created_at": now}])

        migration = MigrationAddJobTimings(version="20261113120000", description="add job timing fields")
        migration.up()
        assert cache.count_documents({field: now for field in CACHE_FIELDS}) == 1
        assert jobs.count_documents({"first_job_created_at": now}) == 1

        migration.down()
        assert [doc.keys() for doc in [*cache.find(), *jobs.find()]] == [{"_id", "dataset"}] * 4

        cache.drop()
        jobs.drop()
