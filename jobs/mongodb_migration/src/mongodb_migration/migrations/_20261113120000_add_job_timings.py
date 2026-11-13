# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import logging

from libcommon.constants import (
    CACHE_COLLECTION_RESPONSES,
    CACHE_MONGOENGINE_ALIAS,
    QUEUE_COLLECTION_JOBS,
    QUEUE_MONGOENGINE_ALIAS,
)
from libcommon.queue.jobs import JobDocument
from libcommon.simple_cache import CachedResponseDocument
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration

CACHE_FIELDS = ["job_created_at", "job_started_at", "revision_created_at", "first_success_at"]
JOB_FIELDS = ["revision_created_at"]


# connection already occurred in the main.py (caveat: we use globals)
class MigrationAddJobTimings(Migration):
    def up(self) -> None:
        # The fields are optional and missing means unknown, so existing documents are left as they are.
        # The migration exists for down(): code without the fields fails to load documents that have them.
        logging.info("Nothing to do: the job timing fields are optional")

    def down(self) -> None:
        logging.info("Remove the job timing fields from the cached results and the jobs")
        get_db(CACHE_MONGOENGINE_ALIAS)[CACHE_COLLECTION_RESPONSES].update_many(
            {}, {"$unset": {field: "" for field in CACHE_FIELDS}}
        )
        get_db(QUEUE_MONGOENGINE_ALIAS)[QUEUE_COLLECTION_JOBS].update_many(
            {}, {"$unset": {field: "" for field in JOB_FIELDS}}
        )

    def validate(self) -> None:
        logging.info("Ensure that a random selection of cached results and jobs can be loaded")

        check_documents(DocCls=CachedResponseDocument, sample_size=10)
        check_documents(DocCls=JobDocument, sample_size=10)
