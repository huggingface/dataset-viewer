# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import types
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypedDict, TypeVar

from bson import ObjectId
from bson.errors import InvalidId
from mongoengine import Document, DoesNotExist, connect
from mongoengine.fields import (
    BooleanField,
    DateTimeField,
    DictField,
    EnumField,
    ObjectIdField,
    StringField,
)
from mongoengine.queryset.queryset import QuerySet

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


def connect_to_database(database: str, host: str) -> None:
    connect(database, alias="cache", host=host)


def get_datetime() -> datetime:
    return datetime.now(timezone.utc)


# cache of the /splits endpoint
class SplitsResponse(Document):
    id = ObjectIdField(db_field="_id", primary_key=True, default=ObjectId)
    dataset_name = StringField(required=True, unique=True)
    http_status = EnumField(HTTPStatus, required=True)
    error_code = StringField(required=False)
    response = DictField(required=True)  # can be an error or a valid content. Not important here.
    details = DictField(required=False)  # can be a detailed error when we don't want to put it in the response.
    stale = BooleanField(required=False, default=False)
    updated_at = DateTimeField(default=get_datetime)
    worker_version = StringField(required=False)
    dataset_git_revision = StringField(required=False)

    meta = {
        "collection": "splitsResponse",
        "db_alias": "cache",
        "indexes": [
            "dataset_name",
            "http_status",
            "stale",
            ("http_status", "error_code"),
            ("error_code", "http_status"),
        ],
    }
    objects = QuerySetManager["SplitsResponse"]()


# cache of the /first-rows endpoint
class FirstRowsResponse(Document):
    id = ObjectIdField(db_field="_id", primary_key=True, default=ObjectId)
    dataset_name = StringField(required=True, unique_with=["config_name", "split_name"])
    config_name = StringField(required=True)
    split_name = StringField(required=True)
    http_status = EnumField(HTTPStatus, required=True)
    error_code = StringField(required=False)
    response = DictField(required=True)  # can be an error or a valid content. Not important here.
    details = DictField(required=False)  # can be a detailed error when we don't want to put it in the response.
    stale = BooleanField(required=False, default=False)
    updated_at = DateTimeField(default=get_datetime)
    worker_version = StringField(required=False)
    dataset_git_revision = StringField(required=False)

    meta = {
        "collection": "firstRowsResponse",
        "db_alias": "cache",
        "indexes": [
            ("dataset_name", "config_name", "split_name"),
            ("dataset_name", "http_status"),
            ("http_status", "dataset_name"),
            # ^ this index (reversed) is used for the "distinct" command to get the names of the valid datasets
            ("http_status", "error_code"),
            ("error_code", "http_status"),
        ],
    }
    objects = QuerySetManager["FirstRowsResponse"]()


AnyResponse = TypeVar("AnyResponse", SplitsResponse, FirstRowsResponse)


# /splits endpoint
# Note: we let the exceptions throw (ie DocumentTooLarge): it's the responsibility of the caller to manage them
def upsert_splits_response(
    dataset_name: str,
    response: Dict,
    http_status: HTTPStatus,
    error_code: Optional[str] = None,
    details: Optional[Dict] = None,
    worker_version: Optional[str] = None,
    dataset_git_revision: Optional[str] = None,
) -> None:
    SplitsResponse.objects(dataset_name=dataset_name).upsert_one(
        http_status=http_status,
        error_code=error_code,
        response=response,
        stale=False,
        details=details,
        updated_at=get_datetime(),
        worker_version=worker_version,
        dataset_git_revision=dataset_git_revision,
    )


def delete_splits_responses(dataset_name: str):
    SplitsResponse.objects(dataset_name=dataset_name).delete()


def mark_splits_responses_as_stale(dataset_name: str):
    SplitsResponse.objects(dataset_name=dataset_name).update(stale=True, updated_at=get_datetime())


class SplitsCacheEntry(TypedDict):
    response: Dict
    http_status: HTTPStatus
    error_code: Optional[str]
    worker_version: Optional[str]
    dataset_git_revision: Optional[str]


# Note: we let the exceptions throw (ie DoesNotExist): it's the responsibility of the caller to manage them
def get_splits_response(dataset_name: str) -> SplitsCacheEntry:
    split_response = SplitsResponse.objects(dataset_name=dataset_name).get()
    return {
        "response": split_response.response,
        "http_status": split_response.http_status,
        "error_code": split_response.error_code,
        "worker_version": split_response.worker_version,
        "dataset_git_revision": split_response.dataset_git_revision,
    }


# /first-rows endpoint
# Note: we let the exceptions throw (ie DocumentTooLarge): it's the responsibility of the caller to manage them
def upsert_first_rows_response(
    dataset_name: str,
    config_name: str,
    split_name: str,
    response: Dict,
    http_status: HTTPStatus,
    error_code: Optional[str] = None,
    details: Optional[Dict] = None,
    worker_version: Optional[str] = None,
    dataset_git_revision: Optional[str] = None,
) -> None:
    FirstRowsResponse.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).upsert_one(
        http_status=http_status,
        error_code=error_code,
        response=response,
        stale=False,
        details=details,
        updated_at=get_datetime(),
        worker_version=worker_version,
        dataset_git_revision=dataset_git_revision,
    )


def delete_first_rows_responses(
    dataset_name: str, config_name: Optional[str] = None, split_name: Optional[str] = None
):
    objects = (
        FirstRowsResponse.objects(dataset_name=dataset_name)
        if config_name is None
        else FirstRowsResponse.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name)
    )
    objects.delete()


def mark_first_rows_responses_as_stale(
    dataset_name: str, config_name: Optional[str] = None, split_name: Optional[str] = None
):
    objects = (
        FirstRowsResponse.objects(dataset_name=dataset_name)
        if config_name is None
        else FirstRowsResponse.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name)
    )
    objects.update(stale=True, updated_at=get_datetime())


# Note: it's the same definition as SplitsCacheEntry
class FirstRowsCacheEntry(TypedDict):
    response: Dict
    http_status: HTTPStatus
    error_code: Optional[str]
    worker_version: Optional[str]
    dataset_git_revision: Optional[str]


# Note: we let the exceptions throw (ie DoesNotExist): it's the responsibility of the caller to manage them
def get_first_rows_response(dataset_name: str, config_name: str, split_name: str) -> FirstRowsCacheEntry:
    first_rows_response = FirstRowsResponse.objects(
        dataset_name=dataset_name, config_name=config_name, split_name=split_name
    ).get()
    return {
        "response": first_rows_response.response,
        "http_status": first_rows_response.http_status,
        "error_code": first_rows_response.error_code,
        "worker_version": first_rows_response.worker_version,
        "dataset_git_revision": first_rows_response.dataset_git_revision,
    }


def get_dataset_first_rows_response_splits(dataset_name: str) -> List[Tuple[str, str, str]]:
    return [
        (firstRowResponse.dataset_name, firstRowResponse.config_name, firstRowResponse.split_name)
        for firstRowResponse in FirstRowsResponse.objects(dataset_name=dataset_name).only(
            "dataset_name", "config_name", "split_name"
        )
    ]


# /valid endpoint


def get_valid_dataset_names() -> List[str]:
    # a dataset is considered valid if:
    # - the /splits response is valid
    candidate_dataset_names = set(SplitsResponse.objects(http_status=HTTPStatus.OK).distinct("dataset_name"))
    # - at least one of the /first-rows responses is valid
    candidate_dataset_names_in_first_rows = set(
        FirstRowsResponse.objects(http_status=HTTPStatus.OK).distinct("dataset_name")
    )

    candidate_dataset_names.intersection_update(candidate_dataset_names_in_first_rows)
    # note that the list is sorted alphabetically for consistency
    return sorted(candidate_dataset_names)


# /is-valid endpoint


def is_dataset_name_valid(dataset_name: str) -> bool:
    # a dataset is considered valid if:
    # - the /splits response is valid
    # - at least one of the /first-rows responses is valid
    valid_split_responses = SplitsResponse.objects(dataset_name=dataset_name, http_status=HTTPStatus.OK).count()
    valid_first_rows_responses = FirstRowsResponse.objects(
        dataset_name=dataset_name, http_status=HTTPStatus.OK
    ).count()
    return (valid_split_responses == 1) and (valid_first_rows_responses > 0)


# admin /metrics endpoint

CountByHttpStatusAndErrorCode = Dict[str, Dict[Optional[str], int]]


def get_entries_count_by_status_and_error_code(entries: QuerySet[AnyResponse]) -> CountByHttpStatusAndErrorCode:
    return {
        str(http_status): {
            error_code: entries(http_status=http_status, error_code=error_code).count()
            for error_code in entries(http_status=http_status).distinct("error_code")
        }
        for http_status in sorted(entries.distinct("http_status"))
    }


def get_splits_responses_count_by_status_and_error_code() -> CountByHttpStatusAndErrorCode:
    return get_entries_count_by_status_and_error_code(SplitsResponse.objects)


def get_first_rows_responses_count_by_status_and_error_code() -> CountByHttpStatusAndErrorCode:
    return get_entries_count_by_status_and_error_code(FirstRowsResponse.objects)


# for scripts


def get_datasets_with_some_error() -> List[str]:
    # - the /splits response is invalid
    candidate_dataset_names = set(SplitsResponse.objects(http_status__ne=HTTPStatus.OK).distinct("dataset_name"))
    # - or one of the /first-rows responses is invalid
    candidate_dataset_names_in_first_rows = set(
        FirstRowsResponse.objects(http_status__ne=HTTPStatus.OK).distinct("dataset_name")
    )

    # note that the list is sorted alphabetically for consistency
    return sorted(candidate_dataset_names.union(candidate_dataset_names_in_first_rows))


# /cache-reports/... endpoints


class SplitsResponseReport(TypedDict):
    dataset: str
    http_status: int
    error_code: Optional[str]
    worker_version: Optional[str]
    dataset_git_revision: Optional[str]


class FirstRowsResponseReport(SplitsResponseReport):
    config: str
    split: str


class CacheReportSplits(TypedDict):
    cache_reports: List[SplitsResponseReport]
    next_cursor: str


class CacheReportFirstRows(TypedDict):
    cache_reports: List[FirstRowsResponseReport]
    next_cursor: str


class InvalidCursor(Exception):
    pass


class InvalidLimit(Exception):
    pass


def get_cache_reports_splits(cursor: str, limit: int) -> CacheReportSplits:
    """
    Get a list of reports about SplitsResponse cache entries, along with the next cursor.
    See https://solovyov.net/blog/2020/api-pagination-design/.
    Args:
        cursor (`str`):
            An opaque string value representing a pointer to a specific SplitsResponse item in the dataset. The
            server returns results after the given pointer.
            An empty string means to start from the beginning.
        limit (strictly positive `int`):
            The maximum number of results.
    Returns:
        [`CacheReportSplits`]: A dict with the list of reports and the next cursor. The next cursor is
        an empty string if there are no more items to be fetched.
    <Tip>
    Raises the following errors:
        - [`~libcache.simple_cache.InvalidCursor`]
          If the cursor is invalid.
        - [`~libcache.simple_cache.InvalidLimit`]
          If the limit is an invalid number.
    </Tip>
    """
    if not cursor:
        queryset = SplitsResponse.objects()
    else:
        try:
            queryset = SplitsResponse.objects(id__gt=ObjectId(cursor))
        except InvalidId as err:
            raise InvalidCursor("Invalid cursor.") from err
    if limit <= 0:
        raise InvalidLimit("Invalid limit.")
    objects = list(
        queryset.order_by("+id")
        .only("id", "dataset_name", "http_status", "error_code", "worker_version", "dataset_git_revision")
        .limit(limit)
    )

    return {
        "cache_reports": [
            {
                "dataset": object.dataset_name,
                "http_status": object.http_status.value,
                "error_code": object.error_code,
                "worker_version": object.worker_version,
                "dataset_git_revision": object.dataset_git_revision,
            }
            for object in objects
        ],
        "next_cursor": "" if len(objects) < limit else str(objects[-1].id),
    }


def get_cache_reports_first_rows(cursor: Optional[str], limit: int) -> CacheReportFirstRows:
    """
    Get a list of reports about FirstRowsResponse cache entries, along with the next cursor.
    See https://solovyov.net/blog/2020/api-pagination-design/.
    Args:
        cursor (`str`):
            An opaque string value representing a pointer to a specific FirstRowsResponse item in the dataset. The
            server returns results after the given pointer.
            An empty string means to start from the beginning.
        limit (strictly positive `int`):
            The maximum number of results.
    Returns:
        [`CacheReportFirstRows`]: A dict with the list of reports and the next cursor. The next cursor is
        an empty string if there are no more items to be fetched.
    <Tip>
    Raises the following errors:
        - [`~libcache.simple_cache.InvalidCursor`]
          If the cursor is invalid.
        - [`~libcache.simple_cache.InvalidLimit`]
          If the limit is an invalid number.
    </Tip>
    """
    if not cursor:
        queryset = FirstRowsResponse.objects()
    else:
        try:
            queryset = FirstRowsResponse.objects(id__gt=ObjectId(cursor))
        except InvalidId as err:
            raise InvalidCursor("Invalid cursor.") from err
    if limit <= 0:
        raise InvalidLimit("Invalid limit.")
    objects = list(
        queryset.order_by("+id")
        .only(
            "id",
            "dataset_name",
            "config_name",
            "split_name",
            "http_status",
            "error_code",
            "worker_version",
            "dataset_git_revision",
        )
        .limit(limit)
    )
    return {
        "cache_reports": [
            {
                "dataset": object.dataset_name,
                "config": object.config_name,
                "split": object.split_name,
                "http_status": object.http_status.value,
                "error_code": object.error_code,
                "worker_version": object.worker_version,
                "dataset_git_revision": object.dataset_git_revision,
            }
            for object in objects
        ],
        "next_cursor": "" if len(objects) < limit else str(objects[-1].id),
    }


class FeaturesResponseReport(TypedDict):
    dataset: str
    config: str
    split: str
    features: Optional[List[Any]]


class CacheReportFeatures(TypedDict):
    cache_reports: List[FeaturesResponseReport]
    next_cursor: str


def get_cache_reports_features(cursor: Optional[str], limit: int) -> CacheReportFeatures:
    """
    Get a list of reports on the features (columns), grouped by splits, along with the next cursor.
    See https://solovyov.net/blog/2020/api-pagination-design/.
    Args:
        cursor (`str`):
            An opaque string value representing a pointer to a specific FirstRowsResponse item in the dataset. The
            server returns results after the given pointer.
            An empty string means to start from the beginning.
        limit (strictly positive `int`):
            The maximum number of results.
    Returns:
        [`CacheReportFeatures`]: A dict with the list of reports and the next cursor. The next cursor is
        an empty string if there are no more items to be fetched.
    <Tip>
    Raises the following errors:
        - [`~libcache.simple_cache.InvalidCursor`]
          If the cursor is invalid.
        - [`~libcache.simple_cache.InvalidLimit`]
          If the limit is an invalid number.
    </Tip>
    """
    if not cursor:
        queryset = FirstRowsResponse.objects()
    else:
        try:
            queryset = FirstRowsResponse.objects(id__gt=ObjectId(cursor))
        except InvalidId as err:
            raise InvalidCursor("Invalid cursor.") from err
    if limit <= 0:
        raise InvalidLimit("Invalid limit.")
    objects = list(
        queryset(response__features__exists=True)
        .order_by("+id")
        .only("id", "dataset_name", "config_name", "split_name", "response.features")
        .limit(limit)
    )
    return {
        "cache_reports": [
            {
                "dataset": object.dataset_name,
                "config": object.config_name,
                "split": object.split_name,
                "features": object.response["features"],
            }
            for object in objects
        ],
        "next_cursor": "" if len(objects) < limit else str(objects[-1].id),
    }


# only for the tests
def _clean_database() -> None:
    SplitsResponse.drop_collection()  # type: ignore
    FirstRowsResponse.drop_collection()  # type: ignore


# explicit re-export
__all__ = ["DoesNotExist"]
