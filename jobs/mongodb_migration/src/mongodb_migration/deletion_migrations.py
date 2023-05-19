# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from mongoengine.connection import get_db

from mongodb_migration.migration import (
    CacheMigration,
    IrreversibleMigrationError,
    MetricsMigration,
    QueueMigration,
)


class MetricsDeletionMigration(MetricsMigration):
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
