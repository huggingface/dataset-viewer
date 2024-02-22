# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from mongoengine.connection import get_db

from mongodb_migration.migration import IrreversibleMigrationError, Migration

STREAMING = "split-first-rows-from-streaming"
PARQUET = "split-first-rows-from-parquet"
MERGED = "split-first-rows"
JOB_RUNNER_VERSION = 4


# connection already occurred in the main.py (caveat: we use globals)
class MigrationMergeSplitFirstRowsResponses(Migration):
    """
    Merge the 'split-first-rows-from-streaming' and 'split-first-rows-from-parquet' responses into one
      'split-first-rows' response. Use "-from-parquet" if successful, otherwise use "-from-streaming".

    The logic is as follows:
      1. remove all the entries with "error_code='ResponseAlreadyComputedError'" for these two kinds
      2. if 'split-first-rows-from-parquet' is the only entry, rename 'kind' to 'split-first-rows'
      3. else, if 'split-first-rows-from-parquet' is a success, rename 'kind' to 'split-first-rows'
        and delete 'split-first-rows-from-streaming'
      4. else, if 'split-first-rows-from-streaming' exists, rename 'kind' to 'split-first-rows'
        and delete 'split-first-rows-from-parquet'

    Stats on 2024-02-21:
    - 192,635 'ResponseAlreadyComputedError' entries will be removed (46% of 417,659 entries)
    - ~36,741 entries will come from 'split-first-rows-from-parquet' (8% of 417,659 entries)
    - ~178,965 entries will come from 'split-first-rows-from-streaming' (43% of 417,659 entries)

    We do it with a loop, even if it's longer than an aggregation, but it's less risky and more readable.
    """

    def up(self) -> None:
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        logging.info(
            "Remove all the entries with 'error_code=ResponseAlreadyComputedError' for 'split-first-rows-from-streaming'"
        )
        db[CACHE_COLLECTION_RESPONSES].delete_many(
            {
                "kind": {"$in": [STREAMING, PARQUET]},
                "error_code": "ResponseAlreadyComputedError",
            }
        )
        logging.info("Update or delete all the 'split-first-rows-from-parquet' responses")
        for parquet_entry in db[CACHE_COLLECTION_RESPONSES].find({"kind": PARQUET}):
            streaming_entry = db[CACHE_COLLECTION_RESPONSES].find_one(
                {
                    "kind": STREAMING,
                    "dataset": parquet_entry["dataset"],
                    "config": parquet_entry["config"],
                    "split": parquet_entry["split"],
                }
            )
            if streaming_entry is None:
                db[CACHE_COLLECTION_RESPONSES].update_one(
                    {"_id": parquet_entry["_id"]}, {"$set": {"kind": MERGED, "job_runner_version": JOB_RUNNER_VERSION}}
                )
            elif parquet_entry["http_status"] == 200:
                db[CACHE_COLLECTION_RESPONSES].update_one(
                    {"_id": parquet_entry["_id"]}, {"$set": {"kind": MERGED, "job_runner_version": JOB_RUNNER_VERSION}}
                )
                db[CACHE_COLLECTION_RESPONSES].delete_one({"_id": streaming_entry["_id"]})
            else:
                db[CACHE_COLLECTION_RESPONSES].update_one(
                    {"_id": streaming_entry["_id"]},
                    {"$set": {"kind": MERGED, "job_runner_version": JOB_RUNNER_VERSION}},
                )
                db[CACHE_COLLECTION_RESPONSES].delete_one({"_id": parquet_entry["_id"]})
        logging.info("Update the remaning 'split-first-rows-from-streaming' entries to 'split-first-rows'")
        db[CACHE_COLLECTION_RESPONSES].update_many(
            {"kind": STREAMING}, {"$set": {"kind": MERGED, "job_runner_version": JOB_RUNNER_VERSION}}
        )

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info(
            "Ensure that no 'split-first-rows-from-streaming' and 'split-first-rows-from-parquet' entries exist"
        )
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        if db[CACHE_COLLECTION_RESPONSES].count_documents({"kind": {"$in": [STREAMING, PARQUET]}}) > 0:
            raise ValueError(
                "Some 'split-first-rows-from-streaming' and 'split-first-rows-from-parquet' entries still exist"
            )
        logging.info("Check 'split-first-rows' responses exist")
        if db[CACHE_COLLECTION_RESPONSES].count_documents({"kind": MERGED}) == 0:
            raise ValueError("No 'split-first-rows' entries exist")
