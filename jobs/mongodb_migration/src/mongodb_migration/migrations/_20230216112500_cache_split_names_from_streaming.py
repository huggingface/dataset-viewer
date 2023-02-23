# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.simple_cache import CachedResponse
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration

db_name = "cache"


# connection already occurred in the main.py (caveat: we use globals)
class MigrationCacheUpdateSplitNames(Migration):
    def up(self) -> None:
        logging.info("Rename cache_kind field from /split-names to /split-names-from-streaming")
        db = get_db(db_name)
        db["cachedResponsesBlue"].update_many(
            {"kind": "/split-names"}, {"$set": {"kind": "/split-names-from-streaming"}}
        )

    def down(self) -> None:
        logging.info("Rollback cache_kind field from /split-names-from-streaming to /split-names")
        db = get_db(db_name)
        db["cachedResponsesBlue"].update_many(
            {"kind": "/split-names-from-streaming"}, {"$set": {"kind": "/split-names"}}
        )

    def validate(self) -> None:
        logging.info("Validate modified documents")

        check_documents(DocCls=CachedResponse, sample_size=10)
