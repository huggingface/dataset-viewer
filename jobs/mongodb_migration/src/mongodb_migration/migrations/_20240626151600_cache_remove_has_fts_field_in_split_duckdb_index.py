# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.simple_cache import CachedResponseDocument
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationRemoveHasFTSFromSplitDuckdbIndexCacheResponse(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        logging.info("Remove 'has_fts' field from cached results of split-duckdb-index")
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {
                "kind": "split-duckdb-index",
                "http_status": 200,
                "content.stemmer": {"$exists": True},
            },
            {
                "$unset": {
                    "content.has_fts": "",
                }
            },
        )

    def down(self) -> None:
        logging.info("Rollback 'has_fts' field for all the cached results")
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {"kind": "split-duckdb-index", "http_status": 200, "content.stemmer": None},
            {
                "$set": {
                    "content.has_fts": False,
                }
            },
        )

        db[CACHE_COLLECTION_RESPONSES].update_many(
            {"kind": "split-duckdb-index", "http_status": 200, "content.stemmer": {"$ne": None}},
            {
                "$set": {
                    "content.has_fts": True,
                }
            },
        )

    def validate(self) -> None:
        logging.info("Ensure that a random selection of cached results have the 'stemmer' field")

        check_documents(DocCls=CachedResponseDocument, sample_size=10)
