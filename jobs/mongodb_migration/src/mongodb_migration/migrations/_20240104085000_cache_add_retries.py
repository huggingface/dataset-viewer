# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.simple_cache import CachedResponseDocument
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationAddRetriesToCacheResponse(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        logging.info("If missing, add the 'retries' field with the default value (0) to the cached results")
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].update_many({"retries": {"$exists": False}}, {"$set": {"retries": 0}})

    def down(self) -> None:
        logging.info("Remove the 'retries' field from all the cached results")
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].update_many({}, {"$unset": {"retries": ""}})

    def validate(self) -> None:
        logging.info("Ensure that a random selection of cached results have the 'retries' field")

        check_documents(DocCls=CachedResponseDocument, sample_size=10)
