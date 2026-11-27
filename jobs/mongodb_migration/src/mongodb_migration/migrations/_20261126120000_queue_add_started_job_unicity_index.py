# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import logging

from libcommon.constants import (
    QUEUE_COLLECTION_JOBS,
    QUEUE_MONGOENGINE_ALIAS,
    QUEUE_STARTED_JOB_UNICITY_INDEX_FIELDS,
    QUEUE_STARTED_JOB_UNICITY_INDEX_FILTER,
    QUEUE_STARTED_JOB_UNICITY_INDEX_NAME,
)
from mongoengine.connection import get_db

from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationQueueAddStartedJobUnicityIndex(Migration):
    def up(self) -> None:
        # Building the index scans the whole collection: it can be created by hand before the deploy, then it's skipped.
        # ensure_indexes() in the workers sees the same name and keys and does nothing.
        collection = get_db(QUEUE_MONGOENGINE_ALIAS)[QUEUE_COLLECTION_JOBS]
        if QUEUE_STARTED_JOB_UNICITY_INDEX_NAME in collection.index_information():
            logging.info(f"The {QUEUE_STARTED_JOB_UNICITY_INDEX_NAME} index already exists on the jobs")
            return
        logging.info(f"Create the {QUEUE_STARTED_JOB_UNICITY_INDEX_NAME} index on the jobs")
        collection.create_index(
            [(field, 1) for field in QUEUE_STARTED_JOB_UNICITY_INDEX_FIELDS],
            name=QUEUE_STARTED_JOB_UNICITY_INDEX_NAME,
            unique=True,
            partialFilterExpression=QUEUE_STARTED_JOB_UNICITY_INDEX_FILTER,
        )

    def down(self) -> None:
        logging.info(f"Remove the {QUEUE_STARTED_JOB_UNICITY_INDEX_NAME} index from the jobs")
        collection = get_db(QUEUE_MONGOENGINE_ALIAS)[QUEUE_COLLECTION_JOBS]
        if QUEUE_STARTED_JOB_UNICITY_INDEX_NAME in collection.index_information():
            collection.drop_index(QUEUE_STARTED_JOB_UNICITY_INDEX_NAME)

    def validate(self) -> None:
        logging.info(f"Check that the {QUEUE_STARTED_JOB_UNICITY_INDEX_NAME} index matches the one of the workers")
        indexes = get_db(QUEUE_MONGOENGINE_ALIAS)[QUEUE_COLLECTION_JOBS].index_information()
        index = indexes.get(QUEUE_STARTED_JOB_UNICITY_INDEX_NAME)
        if (
            index is None
            or [field for field, _ in index["key"]] != QUEUE_STARTED_JOB_UNICITY_INDEX_FIELDS
            or not index.get("unique")
            or index.get("partialFilterExpression") != QUEUE_STARTED_JOB_UNICITY_INDEX_FILTER
        ):
            raise ValueError(f"the {QUEUE_STARTED_JOB_UNICITY_INDEX_NAME} index is missing or differs: {index}")
