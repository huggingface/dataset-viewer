# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.simple_cache import CachedResponseDocument
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationAddPartialToSplitDuckdbIndexCacheResponse(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        logging.info(
            "If missing, add the 'partial', 'num_rows' and 'num_bytes' fields with the default value"
            " (None, None, None) to the cached results of split-duckdb-index"
        )
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {
                "kind": "split-duckdb-index",
                "http_status": 200,
                "content.partial": {"$exists": False},
            },
            {
                "$set": {
                    "content.partial": None,
                    "content.num_rows": None,
                    "content.num_bytes": None,
                }
            },
        )

    def down(self) -> None:
        logging.info("Remove the 'partial', 'num_rows' and 'num_bytes' fields from all the cached results")
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {
                "kind": "split-duckdb-index",
                "http_status": 200,
            },
            {
                "$unset": {
                    "content.partial": "",
                    "content.num_rows": "",
                    "content.num_bytes": "",
                }
            },
        )

    def validate(self) -> None:
        logging.info(
            "Ensure that a random selection of cached results have the 'partial', 'num_rows' and 'num_bytes' fields"
        )

        check_documents(DocCls=CachedResponseDocument, sample_size=10)
