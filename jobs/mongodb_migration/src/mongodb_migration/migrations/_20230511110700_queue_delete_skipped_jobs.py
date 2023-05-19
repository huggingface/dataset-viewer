# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.queue import Job
from mongoengine import Document
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import IrreversibleMigrationError, Migration

status = "skipped"


# connection already occurred in the main.py (caveat: we use globals)
class MigrationDeleteSkippedJobs(Migration):
    def up(self) -> None:
        logging.info(f"Delete jobs with status {status}.")
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].delete_many({"status": status})

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info("Ensure that a random selection of jobs don't have the status {status}")

        def custom_validation(doc: Document) -> None:
            if not isinstance(doc, Job):
                raise ValueError("Document is not a Job")
            if doc.status == status:
                raise ValueError(f"Document has the status {status}")

        check_documents(DocCls=Job, sample_size=10, custom_validation=custom_validation)
