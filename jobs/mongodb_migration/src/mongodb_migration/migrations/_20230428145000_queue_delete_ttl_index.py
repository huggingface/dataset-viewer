# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from typing import Any, List, Mapping

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from mongoengine.connection import get_db

from mongodb_migration.migration import IrreversibleMigrationError, Migration

field_name = "finished_at"


def get_index_names(index_information: Mapping[str, Any], field_name: str) -> List[str]:
    return [
        name
        for name, value in index_information.items()
        if isinstance(value, dict)
        and "expireAfterSeconds" in value
        and "key" in value
        and value["key"] == [(field_name, 1)]
    ]


class MigrationQueueDeleteTTLIndexOnFinishedAt(Migration):
    def up(self) -> None:
        logging.info(
            f"Delete ttl index on field {field_name}. Mongoengine will create it again with a different TTL parameter"
        )

        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        collection = db[QUEUE_COLLECTION_JOBS]
        ttl_index_names = get_index_names(index_information=collection.index_information(), field_name=field_name)
        if len(ttl_index_names) != 1:
            raise ValueError(f"Expected 1 ttl index on field {field_name}, found {len(ttl_index_names)}")
        collection.drop_index(ttl_index_names[0])

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info("Check that the index does not exists anymore")

        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        collection = db[QUEUE_COLLECTION_JOBS]
        ttl_index_names = get_index_names(index_information=collection.index_information(), field_name=field_name)
        if len(ttl_index_names) > 0:
            raise ValueError(f"Found TTL index for field {field_name}")
