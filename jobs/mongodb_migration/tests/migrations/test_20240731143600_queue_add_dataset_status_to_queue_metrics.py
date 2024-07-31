# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.


from libcommon.constants import QUEUE_MONGOENGINE_ALIAS, TYPE_STATUS_AND_DATASET_STATUS_JOB_COUNTS_COLLECTION
from libcommon.queue.dataset_blockages import DATASET_STATUS_BLOCKED, DATASET_STATUS_NORMAL
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20240731143600_queue_add_dataset_status_to_queue_metrics import (
    MigrationAddDatasetStatusToQueueMetrics,
)


def test_queue_add_dataset_status_to_queue_metrics(mongo_host: str) -> None:
    with MongoResource(
        database="test_queue_add_dataset_status_to_queue_metrics",
        host=mongo_host,
        mongoengine_alias=QUEUE_MONGOENGINE_ALIAS,
    ):
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[TYPE_STATUS_AND_DATASET_STATUS_JOB_COUNTS_COLLECTION].insert_many(
            [
                {
                    "job_type": "job_type1",
                    "status": "waiting",
                },
                {"job_type": "job_type2", "status": "waiting", "dataset_status": DATASET_STATUS_BLOCKED},
            ]
        )

        migration = MigrationAddDatasetStatusToQueueMetrics(
            version="20240731143600", description="add 'dataset_status' field to jobs metrics"
        )
        migration.up()

        result = list(db[TYPE_STATUS_AND_DATASET_STATUS_JOB_COUNTS_COLLECTION].find({"job_type": "job_type1"}))
        assert len(result) == 1
        assert result[0]["dataset_status"] == DATASET_STATUS_NORMAL

        result = list(db[TYPE_STATUS_AND_DATASET_STATUS_JOB_COUNTS_COLLECTION].find({"job_type": "job_type2"}))
        assert len(result) == 1
        assert result[0]["dataset_status"] == DATASET_STATUS_BLOCKED

        migration.down()
        result = list(db[TYPE_STATUS_AND_DATASET_STATUS_JOB_COUNTS_COLLECTION].find())
        assert len(result) == 2
        assert "dataset_status" not in result[0]
        assert "dataset_status" not in result[1]

        db[TYPE_STATUS_AND_DATASET_STATUS_JOB_COUNTS_COLLECTION].drop()
