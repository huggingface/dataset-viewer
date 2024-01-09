# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.simple_cache import CachedResponseDocument
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationAddFailedRunsToCacheResponse(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        logging.info(
            "If missing, add the 'failed_runs' field with the default value (0) to the success cached results"
        )
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {"http_status": 200, "failed_runs": {"$exists": False}}, {"$set": {"failed_runs": 0}}
        )
        logging.info("If missing, add the 'failed_runs' field with a default value (1) to the failed cached results")
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {"http_status": {"$ne": 200}, "failed_runs": {"$exists": False}}, {"$set": {"failed_runs": 1}}
        )

    def down(self) -> None:
        logging.info("Remove the 'failed_runs' field from all the cached results")
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].update_many({}, {"$unset": {"failed_runs": ""}})

    def validate(self) -> None:
        logging.info("Ensure that a random selection of cached results have the 'failed_runs' field")

        check_documents(DocCls=CachedResponseDocument, sample_size=10)
