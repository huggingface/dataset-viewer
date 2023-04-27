# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.queue import Job
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration

first_rows = "/first-rows"
split_first_rows_from_streaming = "split-first-rows-from-streaming"


# connection already occurred in the main.py (caveat: we use globals)
class MigrationQueueUpdateFirstRows(Migration):
    def up(self) -> None:
        logging.info(
            f"Rename unicity_id field from Job[{first_rows}][<dataset>][<config>][split] to"
            f" Job[{split_first_rows_from_streaming}][<dataset>][<config>][split] and change type from {first_rows} to"
            f" {split_first_rows_from_streaming}"
        )

        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].update_many(
            {"type": first_rows},
            [
                {
                    "$set": {
                        "unicity_id": {
                            "$replaceOne": {
                                "input": "$unicity_id",
                                "find": f"Job[{first_rows}]",
                                "replacement": f"Job[{split_first_rows_from_streaming}]",
                            }
                        },
                        "type": split_first_rows_from_streaming,
                    }
                },
            ],  # type: ignore
        )

    def down(self) -> None:
        logging.info(
            f"Rename unicity_id field from Job[{split_first_rows_from_streaming}][<dataset>][<config>][split] to"
            f" Job[{first_rows}][<dataset>][<config>][split] and change type from {split_first_rows_from_streaming} to"
            f" {first_rows}"
        )

        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].update_many(
            {"type": split_first_rows_from_streaming},
            [
                {
                    "$set": {
                        "unicity_id": {
                            "$replaceOne": {
                                "input": "$unicity_id",
                                "find": f"Job[{split_first_rows_from_streaming}]",
                                "replacement": f"Job[{first_rows}]",
                            }
                        },
                        "type": split_first_rows_from_streaming,
                    }
                },
            ],  # type: ignore
        )

    def validate(self) -> None:
        logging.info("Validate modified documents")

        check_documents(DocCls=Job, sample_size=10)
