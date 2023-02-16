# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.queue import Job

from mongoengine.connection import get_db
from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration

split_names = "/split-names"
split_names_from_streaming = "/split-names-from-streaming"
db_name = "queue"


# connection already occurred in the main.py (caveat: we use globals)
class MigrationQueueUpdateSplitNames(Migration):
    def up(self) -> None:
        logging.info(
            f"Rename unicity_id field from Job[{split_names}][<dataset>][<config>][<split>] to"
            f" Job[{split_names_from_streaming}][<dataset>][<config>][<split>]"
        )

        for job in Job.objects(type=split_names):
            job.update(unicity_id=f"Job[{split_names_from_streaming}][{job.dataset}][{job.config}][{job.split}]")

        logging.info(f"Rename type field from {split_names} to {split_names_from_streaming}")
        db = get_db("queue")
        db["jobsBlue"].update_many({"type": split_names}, {"$set": {"type": split_names_from_streaming}})

    def down(self) -> None:
        logging.info(
            f"Rename unicity_id field from Job[{split_names_from_streaming}][<dataset>][<config>][<split>] to"
            f" Job[{split_names}][<dataset>][<config>][<split>]"
        )

        for job in Job.objects(type=split_names_from_streaming):
            job.update(unicity_id=f"Job[{split_names}][{job.dataset}][{job.config}][{job.split}]")

        logging.info(f"Rename type field from {split_names_from_streaming} to {split_names}")
        db = get_db("queue")
        db["jobsBlue"].update_many({"type": split_names_from_streaming}, {"$set": {"type": split_names}})

    def validate(self) -> None:
        logging.info("Validate modified documents")

        check_documents(DocCls=Job, sample_size=10)
