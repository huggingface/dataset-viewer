# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.config import ProcessingGraphConfig
from libcommon.constants import (
    DEFAULT_DIFFICULTY,
    QUEUE_COLLECTION_JOBS,
    QUEUE_MONGOENGINE_ALIAS,
)
from libcommon.queue import JobDocument
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationQueueAddDifficultyToJob(Migration):
    def up(self) -> None:
        logging.info("If missing, add the difficulty with a value that depends on the job type, else 50")
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        processing_graph_config = ProcessingGraphConfig()
        for job_type, spec in processing_graph_config.specification.items():
            difficulty = spec.get("difficulty", DEFAULT_DIFFICULTY)
            db[QUEUE_COLLECTION_JOBS].update_many(
                {"type": job_type, "difficulty": {"$exists": False}},
                {"$set": {"difficulty": difficulty}},
            )
        db[QUEUE_COLLECTION_JOBS].update_many(
            {"difficulty": {"$exists": False}},
            {"$set": {"difficulty": DEFAULT_DIFFICULTY}},
        )

    def down(self) -> None:
        logging.info("Remove the difficulty field from all the jobs")
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].update_many({}, {"$unset": {"difficulty": ""}})

    def validate(self) -> None:
        logging.info("Ensure that a random selection of jobs have the 'difficulty' field set")

        check_documents(DocCls=JobDocument, sample_size=10)
