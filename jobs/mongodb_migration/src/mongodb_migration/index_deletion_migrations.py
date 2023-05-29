# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from typing import Any, List, Mapping, Optional, Tuple

from libcommon.constants import (
    CACHE_COLLECTION_RESPONSES,
    CACHE_MONGOENGINE_ALIAS,
    QUEUE_COLLECTION_JOBS,
    QUEUE_MONGOENGINE_ALIAS,
)
from mongoengine.connection import get_db

from mongodb_migration.migration import IrreversibleMigrationError, Migration


class BaseIndexDeletionMigration(Migration):
    def __init__(
        self,
        mongo_engine_alias: str,
        collection_name: str,
        index_definition: List[Tuple[str, int]],
        version: str,
        description: Optional[str] = None,
    ):
        self.mongo_engine_alias = mongo_engine_alias
        self.collection_name = collection_name
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

        db = get_db(self.mongo_engine_alias)
        collection = db[self.collection_name]
        index_names = self.get_index_names(index_information=collection.index_information())
        if len(index_names) < 1:
            raise ValueError(f"Found {len(index_names)} indexes (should be 1): {index_names}.")
        collection.drop_index(index_names[0])

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info("Check that the indexes do not exist anymore")

        db = get_db(self.mongo_engine_alias)
        collection = db[self.collection_name]
        index_names = self.get_index_names(index_information=collection.index_information())
        if len(index_names) > 0:
            raise ValueError(f"Found indexes: {index_names}")


class QueueIndexDeletionMigration(BaseIndexDeletionMigration):
    def __init__(self, version: str, index_definition: List[Tuple[str, int]], description: Optional[str] = None):
        super().__init__(
            version=version,
            description=description,
            index_definition=index_definition,
            mongo_engine_alias=QUEUE_MONGOENGINE_ALIAS,
            collection_name=QUEUE_COLLECTION_JOBS,
        )


class CacheIndexDeletionMigration(BaseIndexDeletionMigration):
    def __init__(self, version: str, index_definition: List[Tuple[str, int]], description: Optional[str] = None):
        super().__init__(
            version=version,
            description=description,
            index_definition=index_definition,
            mongo_engine_alias=CACHE_MONGOENGINE_ALIAS,
            collection_name=CACHE_COLLECTION_RESPONSES,
        )
