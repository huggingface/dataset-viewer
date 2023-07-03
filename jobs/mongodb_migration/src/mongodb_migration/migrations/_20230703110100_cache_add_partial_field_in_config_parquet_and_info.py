# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.simple_cache import CachedResponse
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationAddPartialToCacheResponse(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        logging.info(
            "If missing, add the partial field with the default value (false) to the cached results of"
            " config-parquet-and-info"
        )
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {
                "kind": "config-parquet-and-info",
                "http_status": 200,
                "content.partial": {"$exists": False},
            },
            {"$set": {"content.partial": False}},
        )

    def down(self) -> None:
        logging.info("Remove the partial field from all the cached results")
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {"kind": "config-parquet-and-info", "http_status": 200}, {"$unset": {"content.partial": ""}}
        )

    def validate(self) -> None:
        logging.info("Ensure that a random selection of cached results have the 'partial' field")

        check_documents(DocCls=CachedResponse, sample_size=10)
