# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from mongoengine.connection import get_db

from mongodb_migration.migration import IrreversibleMigrationError, Migration


class MigrationDropCollection(Migration):
    def __init__(self, version: str, description: str, collection_name: str, alias: str):
        super().__init__(version=version, description=description)
        self.collection_name = collection_name
        self.alias = alias

    def up(self) -> None:
        logging.info(f"drop {self.collection_name} collection from {self.alias}")

        db = get_db(self.alias)
        db[self.collection_name].drop()

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info("check that collection does not exist")

        db = get_db(self.alias)
        collections = db.list_collection_names()  # type: ignore
        if self.collection_name in collections:
            raise ValueError(f"found collection with name {self.collection_name}")
