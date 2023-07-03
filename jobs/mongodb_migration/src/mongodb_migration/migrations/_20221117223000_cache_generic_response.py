# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import contextlib
import logging

from libcommon.simple_cache import CachedResponseDocument
from mongoengine.connection import get_db
from pymongo.errors import InvalidName

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration

db_name = "cache"
splitsResponseCollection = "splitsResponse"
firstRowsResponseCollection = "firstRowsResponse"
cachedResponseCollection = "cachedResponsesBlue"
SPLITS_KIND = "/splits"
FIRST_ROWS_KIND = "/first-rows"


# connection already occurred in the main.py (caveat: we use globals)
class MigrationMoveToGenericCachedResponse(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        logging.info(
            f"Create the {cachedResponseCollection} collection, and fill it with the data from splits and first-rows"
        )
        db = get_db(db_name)
        # Copy the data from the previous collections (splitsResponse, firstRowsResponse) to
        # the new generic collection (cachedResponse)
        with contextlib.suppress(InvalidName):
            for splits_response in db[splitsResponseCollection].find():
                if not isinstance(splits_response, dict):
                    # for mypy
                    raise ValueError("splits_response should be a dict")
                db[cachedResponseCollection].insert_one(
                    {
                        "_id": splits_response.get("_id"),
                        "kind": SPLITS_KIND,
                        # ^ "kind" is a new field
                        "dataset": splits_response.get("dataset_name"),
                        "config": None,
                        "split": None,
                        # ^ "config" and "split" are None for kind=/splits
                        "http_status": splits_response.get("http_status"),
                        "error_code": splits_response.get("error_code"),
                        "content": splits_response.get("response"),
                        # ^ "response" field has been renamed to "content"
                        "worker_version": splits_response.get("worker_version"),
                        "dataset_git_revision": splits_response.get("dataset_git_revision"),
                        "details": splits_response.get("details"),
                        "updated_at": splits_response.get("updated_at"),
                        # "stale" field is not used anymore
                    }
                )
        with contextlib.suppress(InvalidName):
            for first_rows_response in db[firstRowsResponseCollection].find():
                if not isinstance(first_rows_response, dict):
                    # for mypy
                    raise ValueError("first_rows_response should be a dict")
                db[cachedResponseCollection].insert_one(
                    {
                        "_id": first_rows_response.get("_id"),
                        "kind": FIRST_ROWS_KIND,
                        # ^ "kind" is a new field
                        "dataset": first_rows_response.get("dataset_name"),
                        "config": first_rows_response.get("config_name"),
                        "split": first_rows_response.get("split_name"),
                        # ^ "config" and "split" are None for kind=/splits
                        "http_status": first_rows_response.get("http_status"),
                        "error_code": first_rows_response.get("error_code"),
                        "content": first_rows_response.get("response"),
                        # ^ "response" field has been renamed to "content"
                        "worker_version": first_rows_response.get("worker_version"),
                        "dataset_git_revision": first_rows_response.get("dataset_git_revision"),
                        "details": first_rows_response.get("details"),
                        "updated_at": first_rows_response.get("updated_at"),
                        # "stale" field is not used anymore
                    }
                )
        # We will not delete the old collections for now. It will be made in a later migration.
        # Also: no need to create indexes on the new collection, mongoengine will do it automatically on the next
        # request.

    def down(self) -> None:
        logging.info(f"Delete the {cachedResponseCollection} collection")
        db = get_db(db_name)
        with contextlib.suppress(InvalidName):
            db[cachedResponseCollection].drop()

    def validate(self) -> None:
        logging.info("Validate the migrated documents")

        check_documents(DocCls=CachedResponseDocument, sample_size=10)

        db = get_db(db_name)
        try:
            splits_responses_count = db[splitsResponseCollection].count_documents({})
        except InvalidName:
            splits_responses_count = 0
        try:
            first_rows_responses_count = db[firstRowsResponseCollection].count_documents({})
        except InvalidName:
            first_rows_responses_count = 0
        cached_responses_count = CachedResponseDocument.objects.count()
        if splits_responses_count + first_rows_responses_count > cached_responses_count:
            raise ValueError(
                f"Some documents are missing in the new collection: splitsResponse ({splits_responses_count}),"
                f" firstRowsResponse ({first_rows_responses_count}), cachedResponseBlue ({cached_responses_count})"
            )
