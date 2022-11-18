# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import types
from datetime import datetime, timezone
from enum import Enum
from http import HTTPStatus
from typing import Generic, Type, TypeVar

from bson import ObjectId
from mongoengine import Document
from mongoengine.connection import get_db
from mongoengine.fields import (
    DateTimeField,
    DictField,
    EnumField,
    ObjectIdField,
    StringField,
)
from mongoengine.queryset.queryset import QuerySet

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


class CacheKind(Enum):
    SPLITS = "/splits"
    FIRST_ROWS = "/first-rows"


db_name = "cache"
splitsResponseCollection = "splitsResponse"
firstRowsResponseCollection = "firstRowsResponse"
cachedResponseCollection = "cachedResponsesBlue"


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
        for splits_response in db[splitsResponseCollection].find():
            if not isinstance(splits_response, dict):
                # for mypy
                raise ValueError("splits_response should be a dict")
            db[cachedResponseCollection].insert_one(
                {
                    "_id": splits_response.get("_id"),
                    "kind": CacheKind.SPLITS.value,
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
                    "dataset_git_version": splits_response.get("dataset_git_version"),
                    "details": splits_response.get("details"),
                    "updated_at": splits_response.get("updated_at"),
                    # "stale" field is not used anymore
                }
            )
        for first_rows_response in db[firstRowsResponseCollection].find():
            if not isinstance(first_rows_response, dict):
                # for mypy
                raise ValueError("first_rows_response should be a dict")
            db[cachedResponseCollection].insert_one(
                {
                    "_id": first_rows_response.get("_id"),
                    "kind": CacheKind.FIRST_ROWS.value,
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
                    "dataset_git_version": first_rows_response.get("dataset_git_version"),
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
        db[cachedResponseCollection].drop()

    def validate(self) -> None:
        logging.info("Validate the migrated documents")

        def custom_validation(doc: CachedResponseSnapshot) -> None:
            if doc.kind not in (CacheKind.SPLITS.value, CacheKind.FIRST_ROWS.value):
                raise ValueError("kind should be /splits or /first-rows")

        check_documents(DocCls=CachedResponseSnapshot, sample_size=10, custom_validation=custom_validation)

        db = get_db(db_name)
        splits_responses_count = db[splitsResponseCollection].count_documents({})
        first_rows_responses_count = db[firstRowsResponseCollection].count_documents({})
        cached_responses_count = CachedResponseSnapshot.objects.count()
        if splits_responses_count + first_rows_responses_count > cached_responses_count:
            raise ValueError(
                f"Some documents are missing in the new collection: splitsResponse ({splits_responses_count}),"
                f" firstRowsResponse ({first_rows_responses_count}), cachedResponseBlue ({cached_responses_count})"
            )


# --- CachedResponseSnapshot ---

# START monkey patching ### hack ###
# see https://github.com/sbdchd/mongo-types#install
U = TypeVar("U", bound=Document)


def no_op(self, x):  # type: ignore
    return self


QuerySet.__class_getitem__ = types.MethodType(no_op, QuerySet)


class QuerySetManager(Generic[U]):
    def __get__(self, instance: object, cls: Type[U]) -> QuerySet[U]:
        return QuerySet(cls, cls._get_collection())


# END monkey patching ### hack ###


def get_datetime() -> datetime:
    return datetime.now(timezone.utc)


# cache of any endpoint
class CachedResponseSnapshot(Document):
    """A response to an endpoint request, cached in the mongoDB database

    Args:
        kind (`str`): The kind of the cached response, identifies the endpoint
        dataset (`str`): The requested dataset.
        config (`str`, optional): The requested config, if any.
        split (`str`, optional): The requested split, if any.
        http_status (`HTTPStatus`): The HTTP status code.
        error_code (`str`, optional): The error code, if any.
        content (`dict`): The content of the cached response. Can be an error or a valid content.
        details (`dict`, optional): Additional details, eg. a detailed error that we don't want to send as a response.
        updated_at (`datetime`): When the cache entry has been last updated.
        worker_version (`str`): The semver version of the worker that cached the response.
        dataset_git_revision (`str`): The commit (of the git dataset repo) used to generate the response.
    """

    id = ObjectIdField(db_field="_id", primary_key=True, default=ObjectId)

    kind = StringField(required=True, unique_with=["dataset", "config", "split"])
    dataset = StringField(required=True)
    config = StringField()
    split = StringField()

    http_status = EnumField(HTTPStatus, required=True)
    error_code = StringField()
    content = DictField(required=True)
    worker_version = StringField()
    dataset_git_revision = StringField()

    details = DictField()
    updated_at = DateTimeField(default=get_datetime)

    meta = {
        "collection": cachedResponseCollection,
        "db_alias": db_name,
        "indexes": [
            ("dataset", "config", "split"),
            ("dataset", "http_status"),
            ("http_status", "dataset"),
            # ^ this index (reversed) is used for the "distinct" command to get the names of the valid datasets
            ("http_status", "error_code"),
            ("dataset", "-updated_at"),
        ],
    }
    objects = QuerySetManager["CachedResponseSnapshot"]()
