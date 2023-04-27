# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
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
            f"Rename unicity_id field from Job[{split_names}][<dataset>][<config>][None] to"
            f" Job[{split_names_from_streaming}][<dataset>][<config>][None] and change type from {split_names} to"
            f" {split_names_from_streaming}"
        )

        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].update_many(
            {"type": split_names},
            [
                {
                    "$set": {
                        "unicity_id": {
                            "$replaceOne": {
                                "input": "$unicity_id",
                                "find": f"Job[{split_names}]",
                                "replacement": f"Job[{split_names_from_streaming}]",
                            }
                        },
                        "type": split_names_from_streaming,
                    }
                },
            ],  # type: ignore
        )

    def down(self) -> None:
        logging.info(
            f"Rename unicity_id field from Job[{split_names_from_streaming}][<dataset>][<config>][None] to"
            f" Job[{split_names}][<dataset>][<config>][None] and change type from {split_names_from_streaming} to"
            f" {split_names}"
        )

        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].update_many(
            {"type": split_names_from_streaming},
            [
                {
                    "$set": {
                        "unicity_id": {
                            "$replaceOne": {
                                "input": "$unicity_id",
                                "find": f"Job[{split_names_from_streaming}]",
                                "replacement": f"Job[{split_names}]",
                            }
                        },
                        "type": split_names_from_streaming,
                    }
                },
            ],  # type: ignore
        )

    def validate(self) -> None:
        logging.info("Validate modified documents")

        check_documents(DocCls=Job, sample_size=10)
