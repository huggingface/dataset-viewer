# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.simple_cache import CachedResponse
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationAddJobRunnerVersionToCacheResponse(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        logging.info("If missing, add 'job_runner_version' field based on 'worker_version' value")
        db = get_db("cache")
        db["cachedResponsesBlue"].update_many(
            {"job_runner_version": {"$exists": False}},
            [
                {
                    "$set": {
                        "job_runner_version": {
                            "$convert": {
                                "input": {"$first": {"$split": ["$worker_version", "."]}},
                                "to": "int",
                                "onError": None,
                                "onNull": None,
                            }
                        }
                    }
                }
            ],  # type: ignore
        )

    def down(self) -> None:
        logging.info("Remove 'job_runner_version' field from all the cached results")
        db = get_db("cache")
        db["cachedResponsesBlue"].update_many({}, {"$unset": {"job_runner_version": ""}})

    def validate(self) -> None:
        logging.info("Ensure that a random selection of cached results have the 'job_runner_version' field")

        check_documents(DocCls=CachedResponse, sample_size=10)
