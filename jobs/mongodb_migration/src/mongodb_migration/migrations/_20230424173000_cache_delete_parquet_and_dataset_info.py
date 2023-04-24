# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.


import logging

from mongoengine.connection import get_db

from mongodb_migration.migration import IrreversibleMigrationError, Migration

cache_kind = "/parquet-and-dataset-info"
db_name = "cache"


class MigrationCacheDeleteParquetAndDatasetInfo(Migration):
    def up(self) -> None:
        logging.info(f"Delete cahe entries of kind {cache_kind}")
        db = get_db(db_name)

        # update existing documents with the old kind
        db["cachedResponsesBlue"].delete_many({"kind": cache_kind})

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info(f"Check that none of the documents has the {cache_kind} kind")

        db = get_db(db_name)
        if db[db_name].count_documents({"kind": cache_kind}):
            raise ValueError(f"Found documents with kind {cache_kind}")
