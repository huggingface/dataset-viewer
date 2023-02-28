# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.simple_cache import CachedResponse
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration

split_names = "/split-names"
split_names_from_streaming = "/split-names-from-streaming"
db_name = "cache"


# connection already occurred in the main.py (caveat: we use globals)
class MigrationCacheUpdateSplitNames(Migration):
    def up(self) -> None:
        logging.info(f"Rename cache_kind field from {split_names} to {split_names_from_streaming}")
        db = get_db(db_name)
        db["cachedResponsesBlue"].update_many({"kind": split_names}, {"$set": {"kind": split_names_from_streaming}})

    def down(self) -> None:
        logging.info(f"Rollback cache_kind field from {split_names_from_streaming} to {split_names}")
        db = get_db(db_name)
        db["cachedResponsesBlue"].update_many({"kind": split_names_from_streaming}, {"$set": {"kind": split_names}})

    def validate(self) -> None:
        logging.info("Validate modified documents")

        check_documents(DocCls=CachedResponse, sample_size=10)
