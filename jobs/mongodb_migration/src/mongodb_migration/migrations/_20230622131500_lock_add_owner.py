# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.constants import QUEUE_COLLECTION_LOCKS, QUEUE_MONGOENGINE_ALIAS
from libcommon.queue import Lock
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationAddOwnerToQueueLock(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        logging.info("If missing, add the owner field with the same value as the field job_id to the locks")
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_LOCKS].update_many(
            {"owner": {"$exists": False}},
            [{"$set": {"owner": "$job_id"}}],  # type: ignore
        )

    def down(self) -> None:
        logging.info("Remove the owner field from all the locks")
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_LOCKS].update_many({}, {"$unset": {"owner": ""}})

    def validate(self) -> None:
        logging.info("Ensure that a random selection of locks have the 'owner' field")

        check_documents(DocCls=Lock, sample_size=10)
