# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.simple_cache import CachedResponseDocument
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationAddTagsToHubCacheCacheResponse(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        logging.info(
            "If missing, add the 'tags' field with the default value ([]) to the success cached results of dataset-hub-cache"
        )
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {"kind": "dataset-hub-cache", "http_status": 200, "tags": {"$exists": False}}, {"$set": {"tags": []}}
        )

    def down(self) -> None:
        logging.info("Remove the 'tags' field from all the cached results of dataset-hub-cache")
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].update_many({"kind": "dataset-hub-cache"}, {"$unset": {"tags": ""}})

    def validate(self) -> None:
        logging.info("Ensure that a random selection of cached results of dataset-hub-cache have the 'tags' field")

        check_documents(DocCls=CachedResponseDocument, sample_size=10)
