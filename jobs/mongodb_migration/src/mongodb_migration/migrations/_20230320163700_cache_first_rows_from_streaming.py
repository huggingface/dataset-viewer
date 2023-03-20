# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.simple_cache import CachedResponse
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration

first_rows = "/first-rows"
first_rows_from_streaming = "first-rows-from-streaming"
first_rows_tmp = "/first-rows-TMP"
db_name = "cache"


# connection already occurred in the main.py (caveat: we use globals)
class MigrationCacheUpdateFirstRows(Migration):
    def up(self) -> None:
        logging.info(f"Rename cache_kind field from {first_rows} to {first_rows_from_streaming}")
        db = get_db(db_name)
        # update existing documents with the new kind (if any) to avoid duplicates (will be deleted later)
        db["cachedResponsesBlue"].update_many({"kind": first_rows_from_streaming}, {"$set": {"kind": first_rows_tmp}})
        # update existing documents with the old kind
        db["cachedResponsesBlue"].update_many({"kind": first_rows}, {"$set": {"kind": first_rows_from_streaming}})
        # delete the duplicates
        db["cachedResponsesBlue"].delete_many({"kind": first_rows_tmp})

    def down(self) -> None:
        logging.info(f"Rollback cache_kind field from {first_rows_from_streaming} to {first_rows}")
        db = get_db(db_name)
        db["cachedResponsesBlue"].update_many({"kind": first_rows_from_streaming}, {"$set": {"kind": first_rows}})

    def validate(self) -> None:
        logging.info("Validate modified documents")

        check_documents(DocCls=CachedResponse, sample_size=10)
