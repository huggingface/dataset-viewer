# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.constants import (
    CACHE_COLLECTION_RESPONSES,
    CACHE_MONGOENGINE_ALIAS,
    METRICS_COLLECTION_CACHE_TOTAL_METRIC,
    METRICS_COLLECTION_JOB_TOTAL_METRIC,
    METRICS_MONGOENGINE_ALIAS,
    QUEUE_COLLECTION_JOBS,
    QUEUE_MONGOENGINE_ALIAS,
)
from libcommon.queue import JobDocument
from libcommon.resources import MongoResource
from libcommon.utils import get_datetime
from mongoengine.connection import get_db

from mongodb_migration.deletion_migrations import (
    CacheDeletionMigration,
    MetricsDeletionMigration,
    MigrationQueueDeleteTTLIndex,
    QueueDeletionMigration,
    get_index_names,
)


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


def test_metrics_deletion_migration(mongo_host: str) -> None:
    step_name = job_type = cache_kind = "step_name"
    with MongoResource(
        database="test_metrics_delete_migration",
        host=mongo_host,
        mongoengine_alias=METRICS_MONGOENGINE_ALIAS,
    ):
        db = get_db(METRICS_MONGOENGINE_ALIAS)
        db[METRICS_COLLECTION_JOB_TOTAL_METRIC].insert_many([{"queue": job_type, "status": "waiting", "total": 0}])
        db[METRICS_COLLECTION_CACHE_TOTAL_METRIC].insert_many([{"kind": cache_kind, "http_status": 400, "total": 0}])
        assert db[METRICS_COLLECTION_JOB_TOTAL_METRIC].find_one(
            {"queue": job_type}
        )  # Ensure there is at least one record to delete
        assert db[METRICS_COLLECTION_CACHE_TOTAL_METRIC].find_one(
            {"kind": cache_kind}
        )  # Ensure there is at least one record to delete

        migration = MetricsDeletionMigration(
            job_type=job_type,
            cache_kind=cache_kind,
            version="20230505180300",
            description=f"delete the queue and cache metrics for step '{step_name}'",
        )
        migration.up()

        assert not db[METRICS_COLLECTION_JOB_TOTAL_METRIC].find_one(
            {"queue": job_type}
        )  # Ensure 0 records after deletion
        assert not db[METRICS_COLLECTION_CACHE_TOTAL_METRIC].find_one(
            {"kind": cache_kind}
        )  # Ensure 0 records after deletion

        db[METRICS_COLLECTION_JOB_TOTAL_METRIC].drop()
        db[METRICS_COLLECTION_CACHE_TOTAL_METRIC].drop()


def test_queue_delete_ttl_index(mongo_host: str) -> None:
    with MongoResource(database="test_queue_delete_ttl_index", host=mongo_host, mongoengine_alias="queue"):
        JobDocument(
            type="test",
            dataset="test",
            revision="test",
            unicity_id="test",
            namespace="test",
            created_at=get_datetime(),
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
