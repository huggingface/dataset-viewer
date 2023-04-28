# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.constants import (
    METRICS_COLLECTION_CACHE_TOTAL_METRIC,
    METRICS_COLLECTION_JOB_TOTAL_METRIC,
    METRICS_MONGOENGINE_ALIAS,
)
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230428193100_metrics_delete_dataset_split_names_from_streaming import (
    MigrationMetricsDeleteDatasetSplitNamesFromStreaming,
)


def test_metrics_delete_dataset_split_names_from_streaming(mongo_host: str) -> None:
    step_name = job_type = cache_kind = "dataset-split-names-from-streaming"
    with MongoResource(
        database="test_metrics_delete_dataset_split_names_from_streaming",
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

        migration = MigrationMetricsDeleteDatasetSplitNamesFromStreaming(
            version="20230428193700",
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
