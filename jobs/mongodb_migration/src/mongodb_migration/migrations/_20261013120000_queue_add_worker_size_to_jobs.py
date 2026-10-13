# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import logging

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.dtos import WorkerSize
from libcommon.queue.jobs import JobDocument
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration

INDEX_NAME = "priority_1_status_1_worker_size_1_created_at_1_dataset_1_namespace_1"
INDEX_KEYS = [
    ("priority", 1),
    ("status", 1),
    ("worker_size", 1),
    ("created_at", 1),
    ("dataset", 1),
    ("namespace", 1),
]


# connection already occurred in the main.py (caveat: we use globals)
class MigrationQueueAddWorkerSizeToJobs(Migration):
    def up(self) -> None:
        # The intervals are the ones of WorkerSizeJobsCountDocument.get_worker_size.
        logging.info("If missing, add the 'worker_size' field to the jobs, derived from 'difficulty'")
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        for worker_size, difficulty_filter in [
            (WorkerSize.light, {"$lte": 40}),
            (WorkerSize.medium, {"$gt": 40, "$lte": 70}),
            (WorkerSize.heavy, {"$gt": 70}),
        ]:
            db[QUEUE_COLLECTION_JOBS].update_many(
                {"worker_size": {"$exists": False}, "difficulty": difficulty_filter},
                {"$set": {"worker_size": worker_size.value}},
            )
        # The workers hint this index by name as soon as they restart: build it here, before them.
        # ensure_indexes() in the workers sees the same name and keys and does nothing.
        logging.info(f"Create the {INDEX_NAME} index on the jobs")
        db[QUEUE_COLLECTION_JOBS].create_index(INDEX_KEYS, name=INDEX_NAME)

    def down(self) -> None:
        logging.info("Remove the 'worker_size' field and its index from the jobs")
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        if INDEX_NAME in db[QUEUE_COLLECTION_JOBS].index_information():
            db[QUEUE_COLLECTION_JOBS].drop_index(INDEX_NAME)
        db[QUEUE_COLLECTION_JOBS].update_many({}, {"$unset": {"worker_size": ""}})

    def validate(self) -> None:
        logging.info("Ensure that a random selection of jobs have the 'worker_size' field")

        check_documents(DocCls=JobDocument, sample_size=10)
