# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.simple_cache import CachedResponse
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationAddProgressToCacheResponse(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        logging.info("If missing, add the cache field with the default value (1.0) to the cached results")
        db = get_db("cache")
        db["cachedResponsesBlue"].update_many({"progress": {"$exists": False}}, {"$set": {"progress": 1.0}})

    def down(self) -> None:
        logging.info("Remove the progress field from all the cached results")
        db = get_db("cache")
        db["cachedResponsesBlue"].update_many({}, {"$unset": {"progress": ""}})

    def validate(self) -> None:
        logging.info("Ensure that a random selection of cached results have the 'progress' field")

        check_documents(DocCls=CachedResponse, sample_size=10)
