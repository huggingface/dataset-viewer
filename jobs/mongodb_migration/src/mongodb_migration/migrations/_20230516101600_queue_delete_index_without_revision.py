# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from typing import Any, List, Mapping

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from mongoengine.connection import get_db

from mongodb_migration.migration import IrreversibleMigrationError, Migration

INDEX_DEFINITION = [("type", 1), ("dataset", 1), ("config", 1), ("split", 1), ("status", 1), ("priority", 1)]


def get_index_names(index_information: Mapping[str, Any]) -> List[str]:
    return [
        name
        for name, value in index_information.items()
        if isinstance(value, dict) and "key" in value and value["key"] == INDEX_DEFINITION
    ]


class MigrationQueueDeleteIndexWithoutRevision(Migration):
    def up(self) -> None:
        logging.info("Delete index.")

        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        collection = db[QUEUE_COLLECTION_JOBS]
        index_names = get_index_names(index_information=collection.index_information())
        if len(index_names) != 1:
            raise ValueError(f"Found {len(index_names)} indexes (should be 1): {index_names}.")
        collection.drop_index(index_names[0])

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info("Check that the indexes do not exist anymore")

        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        collection = db[QUEUE_COLLECTION_JOBS]
        index_names = get_index_names(index_information=collection.index_information())
        if len(index_names) > 0:
            raise ValueError(f"Found indexes: {index_names}")
