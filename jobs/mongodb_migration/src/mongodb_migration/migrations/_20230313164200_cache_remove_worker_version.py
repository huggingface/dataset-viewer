# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.simple_cache import CachedResponse
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import IrreversibleMigration, Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationRemoveWorkerVersionFromCachedResponse(Migration):
    def up(self) -> None:
        logging.info("Removing 'worker_version' field.")
        db = get_db("cache")
        db["cachedResponsesBlue"].update_many({}, {"$unset": {"worker_version": ""}})

    def down(self) -> None:
        raise IrreversibleMigration("This migration does not support rollback")

    def validate(self) -> None:
        logging.info("Ensure that a random selection of cached results don't have 'worker_version' field")

        check_documents(DocCls=CachedResponse, sample_size=10)
