# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.


import logging

from libcommon.constants import (
    METRICS_COLLECTION_CACHE_TOTAL_METRIC,
    METRICS_COLLECTION_JOB_TOTAL_METRIC,
    METRICS_MONGOENGINE_ALIAS,
)
from mongoengine.connection import get_db

from mongodb_migration.migration import IrreversibleMigrationError, Migration

job_type = cache_kind = "dataset-split-names-from-streaming"


class MigrationMetricsDeleteDatasetSplitNamesFromStreaming(Migration):
    def up(self) -> None:
        logging.info(f"Delete job metrics of type {job_type}")

        db = get_db(METRICS_MONGOENGINE_ALIAS)
        db[METRICS_COLLECTION_JOB_TOTAL_METRIC].delete_many({"queue": job_type})
        db[METRICS_COLLECTION_CACHE_TOTAL_METRIC].delete_many({"kind": cache_kind})

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info(f"Check that none of the documents has the {job_type} type or {cache_kind} kind")

        db = get_db(METRICS_MONGOENGINE_ALIAS)
        if db[METRICS_COLLECTION_JOB_TOTAL_METRIC].count_documents({"queue": job_type}):
            raise ValueError(f"Found documents with type {job_type}")
        if db[METRICS_COLLECTION_CACHE_TOTAL_METRIC].count_documents({"kind": cache_kind}):
            raise ValueError(f"Found documents with kind {cache_kind}")
