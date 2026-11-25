# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MIN_DIFFICULTY_INDEX_NAME, CACHE_MONGOENGINE_ALIAS
from mongoengine.connection import get_db

from mongodb_migration.migration import Migration

INDEX_KEYS = [("dataset", 1), ("kind", 1)]


# connection already occurred in the main.py (caveat: we use globals)
class MigrationCacheAddMinDifficultyIndex(Migration):
    def up(self) -> None:
        # Build it here, before the workers: ensure_indexes() in the workers sees the same name and keys and does nothing.
        logging.info(f"Create the {CACHE_MIN_DIFFICULTY_INDEX_NAME} index on the cache")
        get_db(CACHE_MONGOENGINE_ALIAS)[CACHE_COLLECTION_RESPONSES].create_index(
            INDEX_KEYS,
            name=CACHE_MIN_DIFFICULTY_INDEX_NAME,
            partialFilterExpression={"min_difficulty": {"$exists": True}},
        )

    def down(self) -> None:
        logging.info(f"Remove the {CACHE_MIN_DIFFICULTY_INDEX_NAME} index from the cache")
        collection = get_db(CACHE_MONGOENGINE_ALIAS)[CACHE_COLLECTION_RESPONSES]
        if CACHE_MIN_DIFFICULTY_INDEX_NAME in collection.index_information():
            collection.drop_index(CACHE_MIN_DIFFICULTY_INDEX_NAME)

    def validate(self) -> None:
        logging.info(f"Check that the {CACHE_MIN_DIFFICULTY_INDEX_NAME} index exists")
        if (
            CACHE_MIN_DIFFICULTY_INDEX_NAME
            not in get_db(CACHE_MONGOENGINE_ALIAS)[CACHE_COLLECTION_RESPONSES].index_information()
        ):
            raise ValueError(f"the {CACHE_MIN_DIFFICULTY_INDEX_NAME} index is missing")
