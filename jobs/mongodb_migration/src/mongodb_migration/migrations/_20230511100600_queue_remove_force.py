# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.queue import JobDocument
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import IrreversibleMigrationError, Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationRemoveForceFromJob(Migration):
    def up(self) -> None:
        logging.info("Removing 'force' field.")
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].update_many({}, {"$unset": {"force": ""}})

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info("Ensure that a random selection of cached results don't have 'force' field")

        check_documents(DocCls=JobDocument, sample_size=10)
