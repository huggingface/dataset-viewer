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

from mongodb_migration.migrations._20261113120000_add_job_timings import MigrationAddJobTimings


def test_add_job_timings(mongo_host: str) -> None:
    with (
        MongoResource(database="test_add_job_timings_cache", host=mongo_host, mongoengine_alias="cache"),
        MongoResource(database="test_add_job_timings_queue", host=mongo_host, mongoengine_alias="queue"),
    ):
        now = datetime(2026, 11, 13, 12, 0, 0)
        cache = get_db(CACHE_MONGOENGINE_ALIAS)[CACHE_COLLECTION_RESPONSES]
        jobs = get_db(QUEUE_MONGOENGINE_ALIAS)[QUEUE_COLLECTION_JOBS]
        cache.insert_many(
            [
                {"kind": "kind", "dataset": "old", "http_status": 200},
                {
                    "kind": "kind",
                    "dataset": "new",
                    "http_status": 200,
                    "job_created_at": now,
                    "job_started_at": now,
                    "revision_created_at": now,
                    "first_success_at": now,
                },
            ]
        )
        jobs.insert_many(
            [{"type": "type", "dataset": "old"}, {"type": "type", "dataset": "new", "revision_created_at": now}]
        )

        migration = MigrationAddJobTimings(version="20261113120000", description="add job timing fields")
        migration.up()

        old_entry, new_entry = list(cache.find(sort=[("dataset", -1)]))
        assert "job_created_at" not in old_entry
        assert new_entry["first_success_at"] == now
        assert [job.get("revision_created_at") for job in jobs.find(sort=[("dataset", -1)])] == [None, now]

        migration.down()

        for doc in cache.find():
            assert not {"job_created_at", "job_started_at", "revision_created_at", "first_success_at"} & doc.keys()
        for doc in jobs.find():
            assert "revision_created_at" not in doc
        assert cache.count_documents({}) == 2
        assert jobs.count_documents({}) == 2

        cache.drop()
        jobs.drop()
