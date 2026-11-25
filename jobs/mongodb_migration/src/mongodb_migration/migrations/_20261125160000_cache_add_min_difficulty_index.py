# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MIN_DIFFICULTY_INDEX, CACHE_MONGOENGINE_ALIAS
from mongoengine.connection import get_db

from mongodb_migration.migration import Migration

CACHE_MIN_DIFFICULTY_INDEX_NAME = CACHE_MIN_DIFFICULTY_INDEX["name"]


# connection already occurred in the main.py (caveat: we use globals)
class MigrationCacheAddMinDifficultyIndex(Migration):
    def up(self) -> None:
        # Build it here, before the workers: ensure_indexes() in the workers sees the same name and keys and does nothing.
        logging.info(f"Create the {CACHE_MIN_DIFFICULTY_INDEX_NAME} index on the cache")
        get_db(CACHE_MONGOENGINE_ALIAS)[CACHE_COLLECTION_RESPONSES].create_index(
            [(field, 1) for field in CACHE_MIN_DIFFICULTY_INDEX["fields"]],
            name=CACHE_MIN_DIFFICULTY_INDEX_NAME,
            partialFilterExpression=CACHE_MIN_DIFFICULTY_INDEX["partialFilterExpression"],
        )

    def down(self) -> None:
        logging.info(f"Remove the {CACHE_MIN_DIFFICULTY_INDEX_NAME} index from the cache")
        collection = get_db(CACHE_MONGOENGINE_ALIAS)[CACHE_COLLECTION_RESPONSES]
        if CACHE_MIN_DIFFICULTY_INDEX_NAME in collection.index_information():
            collection.drop_index(CACHE_MIN_DIFFICULTY_INDEX_NAME)

    def validate(self) -> None:
        logging.info(f"Check that the {CACHE_MIN_DIFFICULTY_INDEX_NAME} index matches the one of the workers")
        indexes = get_db(CACHE_MONGOENGINE_ALIAS)[CACHE_COLLECTION_RESPONSES].index_information()
        index = indexes.get(CACHE_MIN_DIFFICULTY_INDEX_NAME)
        if (
            index is None
            or [field for field, _ in index["key"]] != CACHE_MIN_DIFFICULTY_INDEX["fields"]
            or index.get("partialFilterExpression") != CACHE_MIN_DIFFICULTY_INDEX["partialFilterExpression"]
        ):
            raise ValueError(f"the {CACHE_MIN_DIFFICULTY_INDEX_NAME} index is missing or differs: {index}")
