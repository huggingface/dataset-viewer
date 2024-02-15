# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.simple_cache import CachedResponseDocument
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationAddPartialToSplitDescriptiveStatisticsCacheResponse(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        logging.info(
            "If missing, add the 'partial' field with the default value None"
            " to the cached results of split-descriptive-statistics job runner"
        )
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        partial_configs_entries = db[CACHE_COLLECTION_RESPONSES].find(
            {
                "kind": "config-parquet",
                "content.partial": True,
            }
        )
        partial_configs = [(entry["dataset"], entry["config"]) for entry in partial_configs_entries]
        stats_successful_entries = db[CACHE_COLLECTION_RESPONSES].find(
            {
                "kind": "split-descriptive-statistics",
                "http_status": 200,
                "content.partial": {"$exists": False},
            }
        )
        partial_stats_successful_ids = [
            entry["_id"]
            for entry in stats_successful_entries
            if (entry["dataset"], entry["config"]) in partial_configs
        ]
        # set partial: false in all successful entries except for those that are partial according to `config`-parquet`
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {
                "_id": {"$nin": partial_stats_successful_ids},
                "kind": "split-descriptive-statistics",
                "http_status": 200,
                "content.partial": {"$exists": False},
            },
            {
                "$set": {
                    "content.partial": False,
                }
            },
        )
        # set partial: true in successful partial entries
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {
                "_id": {"$in": partial_stats_successful_ids},
                "kind": "split-descriptive-statistics",
                "http_status": 200,
                "content.partial": {"$exists": False},
            },
            {
                "$set": {
                    "content.partial": True,
                }
            },
        )

    def down(self) -> None:
        logging.info("Remove the 'partial' field from all the cached results")
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {
                "kind": "split-descriptive-statistics",
            },
            {
                "$unset": {
                    "content.partial": "",
                }
            },
        )

    def validate(self) -> None:
        logging.info("Ensure that a random selection of cached results have the 'partial' field")

        check_documents(DocCls=CachedResponseDocument, sample_size=10)
