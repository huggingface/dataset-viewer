# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.simple_cache import CachedResponseDocument
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import IrreversibleMigrationError, Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationInvalidateSearchAndFilterCacheResponse(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        logging.info("Set search and filter to False for successful cache entries")
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {
                "kind": {
                    "$in": [
                        "dataset-is-valid",
                        "config-is-valid",
                        "split-is-valid",
                    ]
                },
                "http_status": 200,
            },
            {"$set": {"content.search": False, "content.filter": False}},
        )

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info("Ensure that a random selection of cached results have the expected fields")

        check_documents(DocCls=CachedResponseDocument, sample_size=10)
