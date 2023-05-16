# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.constants import (
    CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS, QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
)
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.renaming_migrations import (
    CacheRenamingMigration,
    QueueRenamingMigration,
)


def test_cache_renaming_migration(mongo_host: str) -> None:
    old_kind, new_kind = "/kind-name", "kind-name"
    with MongoResource(database="test_cache_rename_kind", host=mongo_host, mongoengine_alias=CACHE_MONGOENGINE_ALIAS):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].insert_many([{"kind": old_kind, "dataset": "dataset", "http_status": 200}])
        assert db[CACHE_COLLECTION_RESPONSES].find_one(
            {"kind": old_kind}
        )  # Ensure there is at least one record to update

        migration = CacheRenamingMigration(
            cache_kind=old_kind,
            new_cache_kind=new_kind,
            version="20230516165100",
            description=f"update 'kind' field in cache from {old_kind} to {new_kind}",
        )
        migration.up()

        assert not db[CACHE_COLLECTION_RESPONSES].find_one({"kind": old_kind})  # Ensure 0 records with old kind

        assert db[CACHE_COLLECTION_RESPONSES].find_one({"kind": new_kind})

        db[CACHE_COLLECTION_RESPONSES].drop()


def test_queue_renaming_migration(mongo_host: str) -> None:
    old_job, new_job = "/job-name", "job-name"
    with MongoResource(
        database="test_test_queue_renaming_migration", host=mongo_host, mongoengine_alias=QUEUE_MONGOENGINE_ALIAS
    ):
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].insert_many(
            [
                {
                    "type": old_job,
                    "unicity_id": f"Job[{old_job}][dataset][config][split]",
                    "dataset": "dataset",
                    "http_status": 200,
                }
            ]
        )
        assert db[QUEUE_COLLECTION_JOBS].find_one({"type": old_job})  # Ensure there is at least one record to update

        migration = QueueRenamingMigration(
            job_type=old_job,
            new_job_type=new_job,
            version="20230516170300",
            description=f"update 'type' and 'unicity_id' fields in job from {old_job} to {new_job}",
        )
        migration.up()

        assert not db[QUEUE_COLLECTION_JOBS].find_one({"type": old_job})  # Ensure 0 records with old type

        result = db[QUEUE_COLLECTION_JOBS].find_one({"type": new_job})
        assert result
        assert result["unicity_id"] == f"Job[{new_job}][dataset][config][split]"
        db[QUEUE_COLLECTION_JOBS].drop()
