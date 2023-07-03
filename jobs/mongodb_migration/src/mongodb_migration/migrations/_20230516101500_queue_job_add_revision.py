# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.queue import JobDocument
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationQueueAddRevisionToJob(Migration):
    def up(self) -> None:
        logging.info("If missing, add the revision field with the value ('main') to the jobs")
        # Note that setting the value to "main" is a trick, that should avoid deleting the jobs,
        # since we don't know the git revision when the jobs were created.
        # The functions that create jobs in the code will set revision to the commit hash, not to "main" anymore.
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].update_many({"revision": {"$exists": False}}, {"$set": {"revision": "main"}})

    def down(self) -> None:
        logging.info("Remove the revision field from all the jobs")
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].update_many({}, {"$unset": {"revision": ""}})

    def validate(self) -> None:
        logging.info("Ensure that a random selection of jobs have the 'revision' field set")

        check_documents(DocCls=JobDocument, sample_size=10)
