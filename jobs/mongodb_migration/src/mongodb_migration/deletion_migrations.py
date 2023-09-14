# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from collections.abc import Mapping
from typing import Any, Optional

from mongoengine.connection import get_db

from mongodb_migration.migration import (
    BaseQueueMigration,
    CacheMigration,
    IrreversibleMigrationError,
    MetricsMigration,
    QueueMigration,
)


class MetricsDeletionMigration(MetricsMigration):
    def __init__(self, job_type: str, cache_kind: str, version: str, description: Optional[str] = None):
        if not description:
            description = f"delete the queue and cache metrics for step '{job_type}'"
        super().__init__(job_type=job_type, cache_kind=cache_kind, version=version, description=description)

    def up(self) -> None:
        logging.info(f"Delete job metrics of type {self.job_type}")

        db = get_db(self.MONGOENGINE_ALIAS)
        result = db[self.COLLECTION_JOB_TOTAL_METRIC].delete_many({"queue": self.job_type})
        logging.info(f"{result.deleted_count} deleted job metrics")
        result = db[self.COLLECTION_CACHE_TOTAL_METRIC].delete_many({"kind": self.cache_kind})
        logging.info(f"{result.deleted_count} deleted cache metrics")

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info(f"Check that none of the documents has the {self.job_type} type or {self.cache_kind} kind")

        db = get_db(self.MONGOENGINE_ALIAS)
        if db[self.COLLECTION_JOB_TOTAL_METRIC].count_documents({"queue": self.job_type}):
            raise ValueError(f"Found documents with type {self.job_type}")
        if db[self.COLLECTION_CACHE_TOTAL_METRIC].count_documents({"kind": self.cache_kind}):
            raise ValueError(f"Found documents with kind {self.cache_kind}")


class CacheDeletionMigration(CacheMigration):
    def __init__(self, cache_kind: str, version: str, description: Optional[str] = None):
        if not description:
            description = f"delete the cache entries of kind '{cache_kind}'"
        super().__init__(cache_kind=cache_kind, version=version, description=description)

    def up(self) -> None:
        logging.info(f"Delete cache entries of kind {self.cache_kind}")
        db = get_db(self.MONGOENGINE_ALIAS)

        # delete existing documents
        result = db[self.COLLECTION_RESPONSES].delete_many({"kind": self.cache_kind})
        logging.info(f"{result.deleted_count} deleted cache entries")

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info(f"Check that none of the documents has the {self.cache_kind} kind")

        db = get_db(self.MONGOENGINE_ALIAS)
        if db[self.COLLECTION_RESPONSES].count_documents({"kind": self.cache_kind}):
            raise ValueError(f"Found documents with kind {self.cache_kind}")


class QueueDeletionMigration(QueueMigration):
    def __init__(self, job_type: str, version: str, description: Optional[str] = None):
        if not description:
            description = f"delete the jobs of type '{job_type}'"
        super().__init__(job_type=job_type, version=version, description=description)

    def up(self) -> None:
        logging.info(f"Delete jobs of type {self.job_type}")

        db = get_db(self.MONGOENGINE_ALIAS)
        result = db[self.COLLECTION_JOBS].delete_many({"type": self.job_type})
        logging.info(f"{result.deleted_count} deleted jobs")

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info(f"Check that none of the documents has the {self.job_type} type")

        db = get_db(self.MONGOENGINE_ALIAS)
        if db[self.COLLECTION_JOBS].count_documents({"type": self.job_type}):
            raise ValueError(f"Found documents with type {self.job_type}")


def get_index_names(index_information: Mapping[str, Any], field_name: str) -> list[str]:
    return [
        name
        for name, value in index_information.items()
        if isinstance(value, dict)
        and "expireAfterSeconds" in value
        and "key" in value
        and value["key"] == [(field_name, 1)]
    ]


class MigrationQueueDeleteTTLIndex(BaseQueueMigration):
    def __init__(self, version: str, description: str, field_name: str):
        super().__init__(version=version, description=description)
        self.field_name = field_name

    def up(self) -> None:
        logging.info(
            f"Delete ttl index on field {self.field_name}. Mongoengine will create it again with a different TTL"
            " parameter"
        )

        db = get_db(self.MONGOENGINE_ALIAS)
        collection = db[self.COLLECTION_JOBS]
        ttl_index_names = get_index_names(index_information=collection.index_information(), field_name=self.field_name)
        if len(ttl_index_names) != 1:
            raise ValueError(f"Expected 1 ttl index on field {self.field_name}, found {len(ttl_index_names)}")
        collection.drop_index(ttl_index_names[0])

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info("Check that the index does not exists anymore")

        db = get_db(self.MONGOENGINE_ALIAS)
        collection = db[self.COLLECTION_JOBS]
        ttl_index_names = get_index_names(index_information=collection.index_information(), field_name=self.field_name)
        if len(ttl_index_names) > 0:
            raise ValueError(f"Found TTL index for field {self.field_name}")
