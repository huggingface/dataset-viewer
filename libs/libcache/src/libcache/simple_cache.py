# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import types
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Dict, Generic, List, Optional, Tuple, Type, TypedDict, TypeVar

from bson import ObjectId

# from bson.errors import InvalidId
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


# cache of any endpoint
class CachedResponse(Document):
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
        stale (`bool`, optional): Whether the cached response is stale or not.
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
    stale = BooleanField(default=False)
    updated_at = DateTimeField(default=get_datetime)

    meta = {
        "collection": "cachedResponsesBlue",
        "db_alias": "cache",
        "indexes": [
            ("dataset", "config", "split"),
            ("dataset", "http_status"),
            ("http_status", "dataset"),
            # ^ this index (reversed) is used for the "distinct" command to get the names of the valid datasets
            ("http_status", "error_code"),
            ("dataset", "-updated_at"),
        ],
    }
    objects = QuerySetManager["CachedResponse"]()


# Note: we let the exceptions throw (ie DocumentTooLarge): it's the responsibility of the caller to manage them
def upsert_response(
    kind: str,
    dataset: str,
    content: Dict,
    http_status: HTTPStatus,
    config: Optional[str] = None,
    split: Optional[str] = None,
    error_code: Optional[str] = None,
    details: Optional[Dict] = None,
    worker_version: Optional[str] = None,
    dataset_git_revision: Optional[str] = None,
) -> None:
    CachedResponse.objects(kind=kind, dataset=dataset, config=config, split=split).upsert_one(
        content=content,
        http_status=http_status,
        error_code=error_code,
        details=details,
        worker_version=worker_version,
        dataset_git_revision=dataset_git_revision,
        stale=False,
        updated_at=get_datetime(),
    )


def get_objects(
    kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None
) -> QuerySet[CachedResponse]:
    return (
        CachedResponse.objects(kind=kind, dataset=dataset)
        if config is None
        else CachedResponse.objects(kind=kind, dataset=dataset, config=config)
        if split is None
        else CachedResponse.objects(kind=kind, dataset=dataset, config=config, split=split)
    )


def delete_responses(kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None):
    get_objects(kind=kind, dataset=dataset, config=config, split=split).delete()


def mark_responses_as_stale(kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None):
    get_objects(kind=kind, dataset=dataset, config=config, split=split).update(stale=True, updated_at=get_datetime())


class CacheEntry(TypedDict):
    content: Dict
    http_status: HTTPStatus
    error_code: Optional[str]
    worker_version: Optional[str]
    dataset_git_revision: Optional[str]


# Note: we let the exceptions throw (ie DoesNotExist): it's the responsibility of the caller to manage them
def get_response(kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None) -> CacheEntry:
    response = (
        CachedResponse.objects(kind=kind, dataset=dataset, config=config, split=split)
        .only("content", "http_status", "error_code", "worker_version", "dataset_git_revision")
        .get()
    )
    return {
        "content": response.content,
        "http_status": response.http_status,
        "error_code": response.error_code,
        "worker_version": response.worker_version,
        "dataset_git_revision": response.dataset_git_revision,
    }


def get_dataset_response_ids(dataset: str) -> List[Tuple[str, str, Optional[str], Optional[str]]]:
    return [
        (response.kind, response.dataset, response.config, response.split)
        for response in CachedResponse.objects(dataset=dataset).only("kind", "dataset", "config", "split")
    ]


# /valid endpoint

# STILL TO BE MIGRATED
# def get_valid_dataset_names() -> List[str]:
#     # a dataset is considered valid if:
#     # - the /splits response is valid
#     candidate_dataset_names = set(SplitsResponse.objects(http_status=HTTPStatus.OK).distinct("dataset_name"))
#     # - at least one of the /first-rows responses is valid
#     candidate_dataset_names_in_first_rows = set(
#         FirstRowsResponse.objects(http_status=HTTPStatus.OK).distinct("dataset_name")
#     )

#     candidate_dataset_names.intersection_update(candidate_dataset_names_in_first_rows)
#     # note that the list is sorted alphabetically for consistency
#     return sorted(candidate_dataset_names)


# /is-valid endpoint

# STILL TO BE MIGRATED
# def is_dataset_name_valid(dataset_name: str) -> bool:
#     # a dataset is considered valid if:
#     # - the /splits response is valid
#     # - at least one of the /first-rows responses is valid
#     valid_split_responses = SplitsResponse.objects(dataset_name=dataset_name, http_status=HTTPStatus.OK).count()
#     valid_first_rows_responses = FirstRowsResponse.objects(
#         dataset_name=dataset_name, http_status=HTTPStatus.OK
#     ).count()
#     return (valid_split_responses == 1) and (valid_first_rows_responses > 0)


# # admin /metrics endpoint

# CountByHttpStatusAndErrorCode = Dict[str, Dict[Optional[str], int]]


# def get_entries_count_by_status_and_error_code(entries: QuerySet[AnyResponse]) -> CountByHttpStatusAndErrorCode:
#     return {
#         str(http_status): {
#             error_code: entries(http_status=http_status, error_code=error_code).count()
#             for error_code in entries(http_status=http_status).distinct("error_code")
#         }
#         for http_status in sorted(entries.distinct("http_status"))
#     }


# def get_splits_responses_count_by_status_and_error_code() -> CountByHttpStatusAndErrorCode:
#     return get_entries_count_by_status_and_error_code(SplitsResponse.objects)


# def get_first_rows_responses_count_by_status_and_error_code() -> CountByHttpStatusAndErrorCode:
#     return get_entries_count_by_status_and_error_code(FirstRowsResponse.objects)


# # for scripts


# def get_datasets_with_some_error() -> List[str]:
#     # - the /splits response is invalid
#     candidate_dataset_names = set(SplitsResponse.objects(http_status__ne=HTTPStatus.OK).distinct("dataset_name"))
#     # - or one of the /first-rows responses is invalid
#     candidate_dataset_names_in_first_rows = set(
#         FirstRowsResponse.objects(http_status__ne=HTTPStatus.OK).distinct("dataset_name")
#     )

#     # note that the list is sorted alphabetically for consistency
#     return sorted(candidate_dataset_names.union(candidate_dataset_names_in_first_rows))


# # /cache-reports/... endpoints


# class SplitsResponseReport(TypedDict):
#     dataset: str
#     http_status: int
#     error_code: Optional[str]
#     worker_version: Optional[str]
#     dataset_git_revision: Optional[str]


# class FirstRowsResponseReport(SplitsResponseReport):
#     config: str
#     split: str


# class CacheReportSplits(TypedDict):
#     cache_reports: List[SplitsResponseReport]
#     next_cursor: str


# class CacheReportFirstRows(TypedDict):
#     cache_reports: List[FirstRowsResponseReport]
#     next_cursor: str


# class InvalidCursor(Exception):
#     pass


# class InvalidLimit(Exception):
#     pass


# def get_cache_reports_splits(cursor: str, limit: int) -> CacheReportSplits:
#     """
#     Get a list of reports about SplitsResponse cache entries, along with the next cursor.
#     See https://solovyov.net/blog/2020/api-pagination-design/.
#     Args:
#         cursor (`str`):
#             An opaque string value representing a pointer to a specific SplitsResponse item in the dataset. The
#             server returns results after the given pointer.
#             An empty string means to start from the beginning.
#         limit (strictly positive `int`):
#             The maximum number of results.
#     Returns:
#         [`CacheReportSplits`]: A dict with the list of reports and the next cursor. The next cursor is
#         an empty string if there are no more items to be fetched.
#     <Tip>
#     Raises the following errors:
#         - [`~libcache.simple_cache.InvalidCursor`]
#           If the cursor is invalid.
#         - [`~libcache.simple_cache.InvalidLimit`]
#           If the limit is an invalid number.
#     </Tip>
#     """
#     if not cursor:
#         queryset = SplitsResponse.objects()
#     else:
#         try:
#             queryset = SplitsResponse.objects(id__gt=ObjectId(cursor))
#         except InvalidId as err:
#             raise InvalidCursor("Invalid cursor.") from err
#     if limit <= 0:
#         raise InvalidLimit("Invalid limit.")
#     objects = list(
#         queryset.order_by("+id")
#         .only("id", "dataset_name", "http_status", "error_code", "worker_version", "dataset_git_revision")
#         .limit(limit)
#     )

#     return {
#         "cache_reports": [
#             {
#                 "dataset": object.dataset_name,
#                 "http_status": object.http_status.value,
#                 "error_code": object.error_code,
#                 "worker_version": object.worker_version,
#                 "dataset_git_revision": object.dataset_git_revision,
#             }
#             for object in objects
#         ],
#         "next_cursor": "" if len(objects) < limit else str(objects[-1].id),
#     }


# def get_cache_reports_first_rows(cursor: Optional[str], limit: int) -> CacheReportFirstRows:
#     """
#     Get a list of reports about FirstRowsResponse cache entries, along with the next cursor.
#     See https://solovyov.net/blog/2020/api-pagination-design/.
#     Args:
#         cursor (`str`):
#             An opaque string value representing a pointer to a specific FirstRowsResponse item in the dataset. The
#             server returns results after the given pointer.
#             An empty string means to start from the beginning.
#         limit (strictly positive `int`):
#             The maximum number of results.
#     Returns:
#         [`CacheReportFirstRows`]: A dict with the list of reports and the next cursor. The next cursor is
#         an empty string if there are no more items to be fetched.
#     <Tip>
#     Raises the following errors:
#         - [`~libcache.simple_cache.InvalidCursor`]
#           If the cursor is invalid.
#         - [`~libcache.simple_cache.InvalidLimit`]
#           If the limit is an invalid number.
#     </Tip>
#     """
#     if not cursor:
#         queryset = FirstRowsResponse.objects()
#     else:
#         try:
#             queryset = FirstRowsResponse.objects(id__gt=ObjectId(cursor))
#         except InvalidId as err:
#             raise InvalidCursor("Invalid cursor.") from err
#     if limit <= 0:
#         raise InvalidLimit("Invalid limit.")
#     objects = list(
#         queryset.order_by("+id")
#         .only(
#             "id",
#             "dataset_name",
#             "config_name",
#             "split_name",
#             "http_status",
#             "error_code",
#             "worker_version",
#             "dataset_git_revision",
#         )
#         .limit(limit)
#     )
#     return {
#         "cache_reports": [
#             {
#                 "dataset": object.dataset_name,
#                 "config": object.config_name,
#                 "split": object.split_name,
#                 "http_status": object.http_status.value,
#                 "error_code": object.error_code,
#                 "worker_version": object.worker_version,
#                 "dataset_git_revision": object.dataset_git_revision,
#             }
#             for object in objects
#         ],
#         "next_cursor": "" if len(objects) < limit else str(objects[-1].id),
#     }


# class FeaturesResponseReport(TypedDict):
#     dataset: str
#     config: str
#     split: str
#     features: Optional[List[Any]]


# class CacheReportFeatures(TypedDict):
#     cache_reports: List[FeaturesResponseReport]
#     next_cursor: str


# def get_cache_reports_features(cursor: Optional[str], limit: int) -> CacheReportFeatures:
#     """
#     Get a list of reports on the features (columns), grouped by splits, along with the next cursor.
#     See https://solovyov.net/blog/2020/api-pagination-design/.
#     Args:
#         cursor (`str`):
#             An opaque string value representing a pointer to a specific FirstRowsResponse item in the dataset. The
#             server returns results after the given pointer.
#             An empty string means to start from the beginning.
#         limit (strictly positive `int`):
#             The maximum number of results.
#     Returns:
#         [`CacheReportFeatures`]: A dict with the list of reports and the next cursor. The next cursor is
#         an empty string if there are no more items to be fetched.
#     <Tip>
#     Raises the following errors:
#         - [`~libcache.simple_cache.InvalidCursor`]
#           If the cursor is invalid.
#         - [`~libcache.simple_cache.InvalidLimit`]
#           If the limit is an invalid number.
#     </Tip>
#     """
#     if not cursor:
#         queryset = FirstRowsResponse.objects()
#     else:
#         try:
#             queryset = FirstRowsResponse.objects(id__gt=ObjectId(cursor))
#         except InvalidId as err:
#             raise InvalidCursor("Invalid cursor.") from err
#     if limit <= 0:
#         raise InvalidLimit("Invalid limit.")
#     objects = list(
#         queryset(response__features__exists=True)
#         .order_by("+id")
#         .only("id", "dataset_name", "config_name", "split_name", "response.features")
#         .limit(limit)
#     )
#     return {
#         "cache_reports": [
#             {
#                 "dataset": object.dataset_name,
#                 "config": object.config_name,
#                 "split": object.split_name,
#                 "features": object.response["features"],
#             }
#             for object in objects
#         ],
#         "next_cursor": "" if len(objects) < limit else str(objects[-1].id),
#     }


# only for the tests
def _clean_database() -> None:
    CachedResponse.drop_collection()  # type: ignore


# explicit re-export
__all__ = ["DoesNotExist"]
