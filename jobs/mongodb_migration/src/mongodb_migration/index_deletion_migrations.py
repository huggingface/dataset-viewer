# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from typing import Any, List, Mapping, Optional, Tuple

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from mongoengine.connection import get_db

from mongodb_migration.migration import (
    BaseCacheMigration,
    BaseQueueMigration,
    IrreversibleMigrationError,
)


class QueueIndexDeletionMigration(BaseQueueMigration):
    def __init__(self, version: str, index_definition: List[Tuple[str, int]], description: Optional[str] = None):
        self.index_definition = index_definition
        if not description:
            description = f"delete index with {index_definition} definition."
        super().__init__(version=version, description=description)

    def get_index_names(self, index_information: Mapping[str, Any]) -> List[str]:
        return [
            name
            for name, value in index_information.items()
            if isinstance(value, dict) and "key" in value and value["key"] == self.index_definition
        ]

    def up(self) -> None:
        logging.info("Delete index.")

        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        collection = db[QUEUE_COLLECTION_JOBS]
        index_names = self.get_index_names(index_information=collection.index_information())
        if len(index_names) < 1:
            raise ValueError(f"Found {len(index_names)} indexes (should be 1): {index_names}.")
        collection.drop_index(index_names[0])

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info("Check that the indexes do not exist anymore")

        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        collection = db[QUEUE_COLLECTION_JOBS]
        index_names = self.get_index_names(index_information=collection.index_information())
        if len(index_names) > 0:
            raise ValueError(f"Found indexes: {index_names}")


class CacheIndexDeletionMigration(BaseCacheMigration):
    def __init__(self, version: str, index_definition: List[Tuple[str, int]], description: Optional[str] = None):
        self.index_definition = index_definition
        if not description:
            description = f"delete index with {index_definition} definition."
        super().__init__(version=version, description=description)

    def get_index_names(self, index_information: Mapping[str, Any]) -> List[str]:
        return [
            name
            for name, value in index_information.items()
            if isinstance(value, dict) and "key" in value and value["key"] == self.index_definition
        ]

    def up(self) -> None:
        logging.info("Delete index.")

        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        collection = db[QUEUE_COLLECTION_JOBS]
        index_names = self.get_index_names(index_information=collection.index_information())
        if len(index_names) < 1:
            raise ValueError(f"Found {len(index_names)} indexes (should be 1): {index_names}.")
        collection.drop_index(index_names[0])

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info("Check that the indexes do not exist anymore")

        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        collection = db[QUEUE_COLLECTION_JOBS]
        index_names = self.get_index_names(index_information=collection.index_information())
        if len(index_names) > 0:
            raise ValueError(f"Found indexes: {index_names}")
