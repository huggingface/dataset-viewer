# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from collections.abc import Mapping
from typing import Any

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from mongoengine.connection import get_db

from mongodb_migration.migration import IrreversibleMigrationError, Migration

field_name = "force"


def get_index_names(index_information: Mapping[str, Any], field_name: str) -> list[str]:
    return [
        name
        for name, value in index_information.items()
        if isinstance(value, dict)
        and "key" in value
        and any(t[0] == field_name for t in value["key"] if isinstance(t, tuple) and len(t))
    ]


class MigrationQueueDeleteIndexesWithForce(Migration):
    def up(self) -> None:
        logging.info(f"Delete indexes that contain the {field_name} field.")

        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        collection = db[QUEUE_COLLECTION_JOBS]
        index_names = get_index_names(index_information=collection.index_information(), field_name=field_name)
        for index_name in index_names:
            collection.drop_index(index_name)

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info("Check that the indexes do not exist anymore")

        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        collection = db[QUEUE_COLLECTION_JOBS]
        index_names = get_index_names(index_information=collection.index_information(), field_name=field_name)
        if len(index_names) > 0:
            raise ValueError(f"Found indexes for field {field_name}: {index_names}")
