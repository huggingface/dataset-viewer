# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.queue import Job
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationAddPriorityToJob(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        logging.info("If missing, add the priority field with the default value ('normal') to the jobs")
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].update_many({"priority": {"$exists": False}}, {"$set": {"priority": "normal"}})

    def down(self) -> None:
        logging.info("Remove the priority field from all the jobs")
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].update_many({}, {"$unset": {"priority": ""}})

    def validate(self) -> None:
        logging.info("Ensure that a random selection of jobs have the 'priority' field set")

        check_documents(DocCls=Job, sample_size=10)
