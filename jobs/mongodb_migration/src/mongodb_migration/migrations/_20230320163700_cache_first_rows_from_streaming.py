# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.simple_cache import CachedResponse
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration

first_rows = "/first-rows"
split_first_rows_from_streaming = "split-first-rows-from-streaming"
db_name = "cache"


# connection already occurred in the main.py (caveat: we use globals)
class MigrationCacheUpdateFirstRows(Migration):
    def up(self) -> None:
        logging.info(f"Rename cache_kind field from {first_rows} to {split_first_rows_from_streaming}")
        db = get_db(db_name)

        # update existing documents with the old kind
        db["cachedResponsesBlue"].update_many(
            {"kind": first_rows}, {"$set": {"kind": split_first_rows_from_streaming}}
        )

    def down(self) -> None:
        logging.info(f"Rollback cache_kind field from {split_first_rows_from_streaming} to {first_rows}")
        db = get_db(db_name)
        db["cachedResponsesBlue"].update_many(
            {"kind": split_first_rows_from_streaming}, {"$set": {"kind": first_rows}}
        )

    def validate(self) -> None:
        logging.info("Validate modified documents")

        check_documents(DocCls=CachedResponse, sample_size=10)
