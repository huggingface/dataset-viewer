# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from mongoengine.connection import get_db

from mongodb_migration.migration import IrreversibleMigrationError, Migration

ROOT_STEP = "dataset-config-names"


# connection already occurred in the main.py (caveat: we use globals)
class MigrationSetUpdatedAtToOldestStep(Migration):
    """
    After the migrations:
     - '_20240221103200_cache_merge_config_split_names'
     - '_20240221160700_cache_merge_split_first_rows'
    the updated_at field is not accurate anymore, leading to backfilling a lot of them (in the daily cronjob), leading to
    unnecessary jobs.

    To fix that, we take the updated_at value for the root step, and set it to all the steps for the same dataset.
    """

    def up(self) -> None:
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        logging.info("Setting the updated_at value for all the steps to the one of the root step, for each dataset")
        for root_entry in db[CACHE_COLLECTION_RESPONSES].find({"kind": ROOT_STEP}):
            db[CACHE_COLLECTION_RESPONSES].update_many(
                {"dataset": root_entry["dataset"]},
                {"$set": {"updated_at": root_entry["updated_at"]}},
            )

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info("No need to validate.")
