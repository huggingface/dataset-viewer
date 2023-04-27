# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from mongoengine.connection import get_db

from mongodb_migration.migration import IrreversibleMigrationError, Migration

cache_kind = "/splits"


class MigrationCacheDeleteSplits(Migration):
    def up(self) -> None:
        logging.info(f"Delete cahe entries of kind {cache_kind}")
        db = get_db(CACHE_MONGOENGINE_ALIAS)

        # update existing documents with the old kind
        db[CACHE_COLLECTION_RESPONSES].delete_many({"kind": cache_kind})

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info(f"Check that none of the documents has the {cache_kind} kind")

        db = get_db(CACHE_MONGOENGINE_ALIAS)
        if db[CACHE_COLLECTION_RESPONSES].count_documents({"kind": cache_kind}):
            raise ValueError(f"Found documents with kind {cache_kind}")
