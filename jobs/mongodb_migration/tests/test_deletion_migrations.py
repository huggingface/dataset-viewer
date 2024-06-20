# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import pytest
from libcommon.constants import (
    CACHE_COLLECTION_RESPONSES,
    CACHE_MONGOENGINE_ALIAS,
    QUEUE_COLLECTION_JOBS,
    QUEUE_MONGOENGINE_ALIAS,
)
from libcommon.dtos import Status
from libcommon.queue.jobs import JobDocument
from libcommon.resources import MongoResource
from libcommon.utils import get_datetime
from mongoengine.connection import get_db
from pytest import raises

from mongodb_migration.deletion_migrations import (
    CacheDeletionMigration,
    MigrationDeleteJobsByStatus,
    MigrationQueueDeleteTTLIndex,
    MigrationRemoveFieldFromCache,
    MigrationRemoveFieldFromJob,
    QueueDeletionMigration,
    get_index_names,
)
from mongodb_migration.migration import IrreversibleMigrationError


def test_cache_deletion_migration(mongo_host: str) -> None:
    kind = "cache_kind"
    with MongoResource(
        database="test_cache_delete_migration",
        host=mongo_host,
        mongoengine_alias=CACHE_MONGOENGINE_ALIAS,
    ):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].insert_many([{"kind": kind, "dataset": "dataset", "http_status": 200}])
        assert db[CACHE_COLLECTION_RESPONSES].find_one({"kind": kind})  # Ensure there is at least one record to delete

        migration = CacheDeletionMigration(
            cache_kind=kind,
            version="20230505180100",
            description=f"remove cache for kind {kind}",
        )
        migration.up()

        assert not db[CACHE_COLLECTION_RESPONSES].find_one({"kind": kind})  # Ensure 0 records with old kind

        db[CACHE_COLLECTION_RESPONSES].drop()


def test_queue_deletion_migration(mongo_host: str) -> None:
    job_type = "job_type"
    with MongoResource(
        database="test_queue_delete_migration",
        host=mongo_host,
        mongoengine_alias=QUEUE_MONGOENGINE_ALIAS,
    ):
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].insert_many(
            [
                {
                    "type": job_type,
                    "unicity_id": f"{job_type},dataset,config,split",
                    "dataset": "dataset",
                    "revision": "revision",
                    "http_status": 200,
                }
            ]
        )
        assert db[QUEUE_COLLECTION_JOBS].find_one({"type": job_type})  # Ensure there is at least one record to delete

        migration = QueueDeletionMigration(
            job_type=job_type,
            version="20230505180200",
            description=f"remove jobs of type '{job_type}'",
        )
        migration.up()

        assert not db[QUEUE_COLLECTION_JOBS].find_one({"type": job_type})  # Ensure 0 records with old type

        db[QUEUE_COLLECTION_JOBS].drop()


@pytest.mark.skip(reason="obsolete, queue collection does not have 'finished_at' field")
def test_queue_delete_ttl_index(mongo_host: str) -> None:
    with MongoResource(database="test_queue_delete_ttl_index", host=mongo_host, mongoengine_alias="queue"):
        JobDocument(
            type="test",
            dataset="test",
            revision="test",
            unicity_id="test",
            namespace="test",
            created_at=get_datetime(),
            difficulty=50,
        ).save()
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        assert (
            len(get_index_names(db[QUEUE_COLLECTION_JOBS].index_information(), "finished_at")) == 1
        )  # Ensure the TTL index exists

        migration = MigrationQueueDeleteTTLIndex(
            version="20230428145000",
            description="remove ttl index on field 'finished_at'",
            field_name="finished_at",
        )
        migration.up()

        assert (
            len(get_index_names(db[QUEUE_COLLECTION_JOBS].index_information(), "finished_at")) == 0
        )  # Ensure the TTL index does not exist anymore

        db[QUEUE_COLLECTION_JOBS].drop()


def test_queue_delete_jobs_by_status(mongo_host: str) -> None:
    job_type = "job_type"
    with MongoResource(
        database="test_queue_delete_jobs_by_status",
        host=mongo_host,
        mongoengine_alias=QUEUE_MONGOENGINE_ALIAS,
    ):
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        status_list = ["SUCCESS", "ERROR", "CANCELLED"]
        for status in status_list:
            db[QUEUE_COLLECTION_JOBS].insert_one(
                {
                    "type": job_type,
                    "unicity_id": f"{job_type},dataset,config,split",
                    "dataset": "dataset",
                    "revision": "revision",
                    "http_status": 200,
                    "status": status,
                }
            )
        db[QUEUE_COLLECTION_JOBS].insert_one(
            {
                "type": job_type,
                "unicity_id": f"{job_type},dataset,config,split",
                "dataset": "dataset",
                "revision": "revision",
                "http_status": 200,
                "status": Status.STARTED,
            }
        )
        assert db[QUEUE_COLLECTION_JOBS].count_documents({"status": {"$in": status_list}}) == 3
        assert db[QUEUE_COLLECTION_JOBS].count_documents({"status": Status.STARTED}) == 1

        migration = MigrationDeleteJobsByStatus(
            status_list=status_list,
            version="20231201074900",
            description=f"remove jobs with status '{status_list}'",
        )
        migration.up()

        assert db[QUEUE_COLLECTION_JOBS].count_documents({"status": {"$in": status_list}}) == 0
        assert db[QUEUE_COLLECTION_JOBS].count_documents({"status": Status.STARTED}) == 1

        with raises(IrreversibleMigrationError):
            migration.down()

        db[QUEUE_COLLECTION_JOBS].drop()


def test_queue_remove_field(mongo_host: str) -> None:
    with MongoResource(database="test_queue_remove_field", host=mongo_host, mongoengine_alias="queue"):
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        field_name = "finished_at"
        db[QUEUE_COLLECTION_JOBS].insert_one(
            {"type": "test", "dataset": "dataset", "force": True, field_name: get_datetime()}
        )

        result = db[QUEUE_COLLECTION_JOBS].find_one({"dataset": "dataset"})
        assert result
        assert field_name in result

        migration = MigrationRemoveFieldFromJob(
            field_name=field_name, version="20231201112600", description=f"remove '{field_name}' field from queue"
        )
        migration.up()
        result = db[QUEUE_COLLECTION_JOBS].find_one({"dataset": "dataset"})
        assert result
        assert field_name not in result

        with raises(IrreversibleMigrationError):
            migration.down()
        db[QUEUE_COLLECTION_JOBS].drop()


def test_cache_remove_field(mongo_host: str) -> None:
    with MongoResource(database="test_cache_remove_field", host=mongo_host, mongoengine_alias="cache"):
        field_name = "retries"
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].insert_many(
            [
                {
                    "kind": "/splits",
                    "dataset": "dataset",
                    "http_status": 501,
                    field_name: 2,
                }
            ]
        )

        result = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": "dataset"})
        assert result
        assert field_name in result

        migration = MigrationRemoveFieldFromCache(
            version="20240109155600", description="remove 'retries' field from cache", field_name=field_name
        )
        migration.up()
        result = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": "dataset"})
        assert result
        assert field_name not in result

        with raises(IrreversibleMigrationError):
            migration.down()
        db[CACHE_COLLECTION_RESPONSES].drop()
