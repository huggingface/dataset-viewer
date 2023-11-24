# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.queue import JobDocument
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationQueueAddPenalizationToJob(Migration):
    def up(self) -> None:
        logging.info("If missing, add a default value for 'penalization'")
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].update_many(
            {"penalization": {"$exists": False}},
            {"$set": {"penalization": 0}},
        )

    def down(self) -> None:
        logging.info("Remove the 'penalization' field from all the jobs")
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].update_many({}, {"$unset": {"penalization": ""}})

    def validate(self) -> None:
        logging.info("Ensure that a random selection of jobs have the 'penalization' field set")

        check_documents(DocCls=JobDocument, sample_size=10)
