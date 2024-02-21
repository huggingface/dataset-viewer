# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from mongoengine.connection import get_db

from mongodb_migration.migration import IrreversibleMigrationError, Migration

STREAMING = "config-split-names-from-streaming"
INFO = "config-split-names-from-info"
MERGED = "config-split-names"


# connection already occurred in the main.py (caveat: we use globals)
class MigrationMergeConfigSplitNamesResponses(Migration):
    """
    Merge the 'config-split-names-from-streaming' and 'config-split-names-from-info' responses into one
      'config-split-names' response. Use "-from-info" if successful, otherwise use "-from-streaming".

    The logic is as follows:
      1. remove all the entries with "error_code='ResponseAlreadyComputedError'" for these two kinds
      2. if 'config-split-names-from-info' is the only entry, rename 'kind' to 'config-split-names'
      3. else, if 'config-split-names-from-info' is a success, rename 'kind' to 'config-split-names'
        and delete 'config-split-names-from-streaming'
      2. else, if 'config-split-names-from-streaming' exists, rename 'kind' to 'config-split-names'
        and delete 'config-split-names-from-info'

    Stats on 2024-02-21:
    1. 122,357 entries will be removed (47% of 256,754 entries)
    2. 124,748 entries will come from 'config-split-names-from-streaming' (48% of 256,754 entries)
    2. 6,404 entries will come from 'config-split-names-from-info' (2% of 256,754 entries)

    We do it with a loop, even if it's longer than an aggregation, but it's less risky and more readable.
    """

    def up(self) -> None:
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        logging.info(
            "Remove all the entries with 'error_code=ResponseAlreadyComputedError' for 'config-split-names-from-streaming'"
        )
        db[CACHE_COLLECTION_RESPONSES].delete_many(
            {
                "kind": {"$in": [STREAMING, INFO]},
                "error_code": "ResponseAlreadyComputedError",
            }
        )
        logging.info("Update or delete all the 'config-split-names-from-info' responses")
        for info_entry in db[CACHE_COLLECTION_RESPONSES].find({"kind": INFO}):
            streaming_entry = db[CACHE_COLLECTION_RESPONSES].find_one(
                {
                    "kind": STREAMING,
                    "dataset": info_entry["dataset"],
                    "config": info_entry["config"],
                }
            )
            if streaming_entry is None:
                db[CACHE_COLLECTION_RESPONSES].update_one({"_id": info_entry["_id"]}, {"$set": {"kind": MERGED}})
            elif info_entry["http_status"] == 200:
                db[CACHE_COLLECTION_RESPONSES].update_one({"_id": info_entry["_id"]}, {"$set": {"kind": MERGED}})
                db[CACHE_COLLECTION_RESPONSES].delete_one({"_id": streaming_entry["_id"]})
            else:
                db[CACHE_COLLECTION_RESPONSES].update_one({"_id": streaming_entry["_id"]}, {"$set": {"kind": MERGED}})
                db[CACHE_COLLECTION_RESPONSES].delete_one({"_id": info_entry["_id"]})
        logging.info("Update the remaning 'config-split-names-from-streaming' entries to 'config-split-names'")
        db[CACHE_COLLECTION_RESPONSES].update_many({"kind": STREAMING}, {"$set": {"kind": MERGED}})

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info(
            "Ensure that no 'config-split-names-from-streaming' and 'config-split-names-from-info' entries exist"
        )
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        if db[CACHE_COLLECTION_RESPONSES].count({"kind": {"$in": [STREAMING, INFO]}}) > 0:
            raise ValueError(
                "Some 'config-split-names-from-streaming' and 'config-split-names-from-info' entries still exist"
            )
        logging.info("Check 'config-split-names' responses exist")
        if db[CACHE_COLLECTION_RESPONSES].count({"kind": MERGED}) == 0:
            raise ValueError("No 'config-split-names' entries exist")
