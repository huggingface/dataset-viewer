# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
import logging
from typing import Any

from libcommon.constants import (
    CACHE_COLLECTION_RESPONSES,
    CACHE_MONGOENGINE_ALIAS,
    METRICS_COLLECTION_CACHE_TOTAL_METRIC,
    METRICS_COLLECTION_JOB_TOTAL_METRIC,
    METRICS_MONGOENGINE_ALIAS,
    QUEUE_COLLECTION_JOBS,
    QUEUE_MONGOENGINE_ALIAS,
)
from mongoengine.connection import get_db

from mongodb_migration.migration import IrreversibleMigrationError, Migration


class DeleteMetricsMigration(Migration):
    _MONGOENGINE_ALIAS = METRICS_MONGOENGINE_ALIAS
    _COLLECTION_JOB_TOTAL_METRIC = METRICS_COLLECTION_JOB_TOTAL_METRIC
    _COLLECTION_CACHE_TOTAL_METRIC = METRICS_COLLECTION_CACHE_TOTAL_METRIC

    def __init__(self, job_type: str, cache_kind: str, *args: Any, **kwargs: Any):
        self.job_type = job_type
        self.cache_kind = cache_kind
        super().__init__(*args, **kwargs)

    def up(self) -> None:
        logging.info(f"Delete job metrics of type {self.job_type}")

        db = get_db(self._MONGOENGINE_ALIAS)
        db[self._COLLECTION_JOB_TOTAL_METRIC].delete_many({"queue": self.job_type})
        db[self._COLLECTION_CACHE_TOTAL_METRIC].delete_many({"kind": self.cache_kind})

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info(f"Check that none of the documents has the {self.job_type} type or {self.cache_kind} kind")

        db = get_db(self._MONGOENGINE_ALIAS)
        if db[self._COLLECTION_JOB_TOTAL_METRIC].count_documents({"queue": self.job_type}):
            raise ValueError(f"Found documents with type {self.job_type}")
        if db[self._COLLECTION_CACHE_TOTAL_METRIC].count_documents({"kind": self.cache_kind}):
            raise ValueError(f"Found documents with kind {self.cache_kind}")


class DeleteCacheMigration(Migration):
    _MONGOENGINE_ALIAS = CACHE_MONGOENGINE_ALIAS
    _COLLECTION_RESPONSES = CACHE_COLLECTION_RESPONSES

    def __init__(self, cache_kind: str, *args: Any, **kwargs: Any):
        self.cache_kind = cache_kind
        super().__init__(*args, **kwargs)

    def up(self) -> None:
        logging.info(f"Delete cache entries of kind {self.cache_kind}")
        db = get_db(self._MONGOENGINE_ALIAS)

        # delete existing documents
        db[self._COLLECTION_RESPONSES].delete_many({"kind": self.cache_kind})

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info(f"Check that none of the documents has the {self.cache_kind} kind")

        db = get_db(self._MONGOENGINE_ALIAS)
        if db[self._COLLECTION_RESPONSES].count_documents({"kind": self.cache_kind}):
            raise ValueError(f"Found documents with kind {self.cache_kind}")


class DeleteQueueMigration(Migration):
    _MONGOENGINE_ALIAS = QUEUE_MONGOENGINE_ALIAS
    _COLLECTION_JOBS = QUEUE_COLLECTION_JOBS

    def __init__(self, job_type: str, *args: Any, **kwargs: Any):
        self.job_type = job_type
        super().__init__(*args, **kwargs)

    def up(self) -> None:
        logging.info(f"Delete jobs of type {self.job_type}")

        db = get_db(self._MONGOENGINE_ALIAS)
        db[self._COLLECTION_JOBS].delete_many({"type": self.job_type})

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info(f"Check that none of the documents has the {self.job_type} type")

        db = get_db(self._MONGOENGINE_ALIAS)
        if db[self._COLLECTION_JOBS].count_documents({"type": self.job_type}):
            raise ValueError(f"Found documents with type {self.job_type}")
