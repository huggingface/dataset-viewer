# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import logging

from libcommon.constants import QUEUE_MONGOENGINE_ALIAS, TYPE_STATUS_AND_DATASET_STATUS_JOB_COUNTS_COLLECTION
from libcommon.queue.dataset_blockages import DATASET_STATUS_NORMAL
from libcommon.queue.metrics import JobTotalMetricDocument
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationAddDatasetStatusToQueueMetrics(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        logging.info("If missing, add the 'dataset_status' field with the default value 'normal' to the jobs metrics")
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[TYPE_STATUS_AND_DATASET_STATUS_JOB_COUNTS_COLLECTION].update_many(
            {"dataset_status": {"$exists": False}}, {"$set": {"dataset_status": DATASET_STATUS_NORMAL}}
        )

    def down(self) -> None:
        logging.info("Remove the 'dataset_status' field from all the jobs metrics")
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[TYPE_STATUS_AND_DATASET_STATUS_JOB_COUNTS_COLLECTION].update_many({}, {"$unset": {"dataset_status": ""}})

    def validate(self) -> None:
        logging.info("Ensure that a random selection of jobs metrics have the 'dataset_status' field")

        check_documents(DocCls=JobTotalMetricDocument, sample_size=10)
