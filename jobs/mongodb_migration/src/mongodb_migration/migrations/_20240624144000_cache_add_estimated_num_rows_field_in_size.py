# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.simple_cache import CachedResponseDocument
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationAddEstimatedNumRowsToSizeCacheResponse(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        logging.info(
            "If missing, add the 'estimated_num_rows' field with the default value"
            " None to the cached results of dataset-size and config-size"
        )
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {
                "kind": "config-size",
                "http_status": 200,
                "content.size.config.estimated_num_rows": {"$exists": False},
            },
            {
                "$set": {
                    "content.size.config.estimated_num_rows": None,
                    "content.size.splits.$[].estimated_num_rows": None,
                }
            },
        )
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {
                "kind": "dataset-size",
                "http_status": 200,
                "content.size.dataset.estimated_num_rows": {"$exists": False},
            },
            {
                "$set": {
                    "content.size.dataset.estimated_num_rows": None,
                    "content.size.configs.$[].estimated_num_rows": None,
                    "content.size.splits.$[].estimated_num_rows": None,
                }
            },
        )

    def down(self) -> None:
        logging.info("Remove the 'config-size' field from all the cached results")
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {
                "kind": "config-size",
                "http_status": 200,
            },
            {
                "$unset": {
                    "content.size.config.estimated_num_rows": "",
                    "content.size.splits.$[].estimated_num_rows": "",
                }
            },
        )
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {
                "kind": "dataset-size",
                "http_status": 200,
            },
            {
                "$unset": {
                    "content.size.dataset.estimated_num_rows": "",
                    "content.size.configs.$[].estimated_num_rows": "",
                    "content.size.splits.$[].estimated_num_rows": "",
                }
            },
        )

    def validate(self) -> None:
        logging.info("Ensure that a random selection of cached results have the 'estimated_num_rows' field")

        check_documents(DocCls=CachedResponseDocument, sample_size=10)
