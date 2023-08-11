# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_METRICS_COLLECTION, METRICS_MONGOENGINE_ALIAS
from mongoengine.connection import get_db

from mongodb_migration.migration import IrreversibleMigrationError, Migration


class MigrationDropCacheMetricsCollection(Migration):
    def up(self) -> None:
        # drop collection from metrics database, it will be recreated in cache database
        logging.info(f"drop {CACHE_METRICS_COLLECTION} collection from {METRICS_MONGOENGINE_ALIAS}")

        db = get_db(METRICS_MONGOENGINE_ALIAS)
        db[CACHE_METRICS_COLLECTION].drop()

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info("check that collection does not exist")

        db = get_db(METRICS_MONGOENGINE_ALIAS)
        collections = db.list_collection_names()  # type: ignore
        if CACHE_METRICS_COLLECTION in collections:
            raise ValueError(f"found collection with name {CACHE_METRICS_COLLECTION}")
