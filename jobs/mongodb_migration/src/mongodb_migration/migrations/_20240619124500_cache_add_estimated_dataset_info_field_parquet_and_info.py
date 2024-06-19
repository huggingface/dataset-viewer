# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.simple_cache import CachedResponseDocument
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationAddEstimatedDatasetInfoToParquetAndInfoCacheResponse(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        logging.info(
            "If missing, add the 'estimated_dataset_info' field with the default value"
            " None to the cached results of config-parquet-and-info"
        )
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {
                "kind": "config-parquet-and-info",
                "http_status": 200,
                "content.estimated_dataset_info": {"$exists": False},
            },
            {
                "$set": {
                    "content.estimated_dataset_info": None,
                }
            },
        )

    def down(self) -> None:
        logging.info("Remove the 'estimated_dataset_info' field from all the cached results")
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {
                "kind": "config-parquet-and-info",
                "http_status": 200,
            },
            {
                "$unset": {
                    "content.estimated_dataset_info": "",
                }
            },
        )

    def validate(self) -> None:
        logging.info("Ensure that a random selection of cached results have the 'estimated_dataset_info field")

        check_documents(DocCls=CachedResponseDocument, sample_size=10)
