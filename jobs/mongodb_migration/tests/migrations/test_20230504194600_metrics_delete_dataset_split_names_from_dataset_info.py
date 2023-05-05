# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.


from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.delete_migration import DeleteMetricsMigration

METRICS_MONGOENGINE_ALIAS = DeleteMetricsMigration.MONGOENGINE_ALIAS
METRICS_COLLECTION_JOB_TOTAL_METRIC = DeleteMetricsMigration.COLLECTION_JOB_TOTAL_METRIC
METRICS_COLLECTION_CACHE_TOTAL_METRIC = DeleteMetricsMigration.COLLECTION_CACHE_TOTAL_METRIC


def test_metrics_delete_dataset_split_names_from_dataset_info(mongo_host: str) -> None:
    step_name = job_type = cache_kind = "dataset-split-names-from-dataset-info"
    with MongoResource(
        database="test_metrics_delete_dataset_split_names_from_dataset_info",
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

        migration = DeleteMetricsMigration(
            job_type=job_type,
            cache_kind=cache_kind,
            version="20230504195700",
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
