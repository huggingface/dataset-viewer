# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import types
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Any, Generic, List, Mapping, Optional, Set, Type, TypedDict, TypeVar

from bson import ObjectId
from bson.errors import InvalidId
from mongoengine import Document, DoesNotExist, connect
from mongoengine.fields import (
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


def connect_to_cache_database(database: str, host: str) -> None:
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
        "collection": "cachedResponsesBlue",
        "db_alias": "cache",
        "indexes": [
            ("kind", "dataset", "config", "split"),
            ("dataset", "kind", "http_status"),
            ("kind", "http_status", "dataset"),
            ("kind", "http_status", "error_code"),
            ("kind", "id"),
        ],
    }
    objects = QuerySetManager["CachedResponse"]()


# Fix issue with mongoengine: https://github.com/MongoEngine/mongoengine/issues/1242#issuecomment-810501601
# mongoengine automatically sets "config" and "splits" as required fields, because they are listed in the unique_with
# field of the "kind" field. But it's an error, since unique indexes (which are used to enforce unique_with) accept
# null values, see https://www.mongodb.com/docs/v5.0/core/index-unique/#unique-index-and-missing-field.
CachedResponse.config.required = False  # type: ignore
CachedResponse.split.required = False  # type: ignore


# Note: we let the exceptions throw (ie DocumentTooLarge): it's the responsibility of the caller to manage them
def upsert_response(
    kind: str,
    dataset: str,
    content: Mapping[str, Any],
    http_status: HTTPStatus,
    config: Optional[str] = None,
    split: Optional[str] = None,
    error_code: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
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
        updated_at=get_datetime(),
    )


def delete_response(
    kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None
) -> Optional[int]:
    return CachedResponse.objects(kind=kind, dataset=dataset, config=config, split=split).delete()


def delete_dataset_responses(dataset: str) -> Optional[int]:
    return CachedResponse.objects(dataset=dataset).delete()


class CacheEntryWithoutContent(TypedDict):
    http_status: HTTPStatus
    error_code: Optional[str]
    worker_version: Optional[str]
    dataset_git_revision: Optional[str]


# Note: we let the exceptions throw (ie DoesNotExist): it's the responsibility of the caller to manage them
def get_response_without_content(
    kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None
) -> CacheEntryWithoutContent:
    response = (
        CachedResponse.objects(kind=kind, dataset=dataset, config=config, split=split)
        .only("http_status", "error_code", "worker_version", "dataset_git_revision")
        .get()
    )
    return {
        "http_status": response.http_status,
        "error_code": response.error_code,
        "worker_version": response.worker_version,
        "dataset_git_revision": response.dataset_git_revision,
    }


class CacheEntry(CacheEntryWithoutContent):
    content: Mapping[str, Any]


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


class ResponseId(TypedDict):
    kind: str
    dataset: str
    config: Optional[str]
    split: Optional[str]


def get_dataset_response_ids(dataset: str) -> List[ResponseId]:
    return [
        {
            "kind": response.kind,
            "dataset": response.dataset,
            "config": response.config,
            "split": response.split,
        }
        for response in CachedResponse.objects(dataset=dataset).only("kind", "dataset", "config", "split")
    ]


def get_valid_datasets(kind: str) -> Set[str]:
    return set(CachedResponse.objects(kind=kind, http_status=HTTPStatus.OK).distinct("dataset"))


def get_validity_by_kind(dataset: str) -> Mapping[str, bool]:
    # TODO: rework with aggregate
    entries = CachedResponse.objects(dataset=dataset).only("kind", "http_status")
    return {
        str(kind): entries(kind=kind, http_status=HTTPStatus.OK).first() is not None
        for kind in sorted(entries.distinct("kind"))
    }


# admin /metrics endpoint


class CountEntry(TypedDict):
    kind: str
    http_status: int
    error_code: Optional[str]
    count: int


def get_responses_count_by_kind_status_and_error_code() -> List[CountEntry]:
    # TODO: rework with aggregate
    # see
    # https://stackoverflow.com/questions/47301829/mongodb-distinct-count-for-combination-of-two-fields?noredirect=1&lq=1#comment81555081_47301829
    # and https://docs.mongoengine.org/guide/querying.html#mongodb-aggregation-api
    entries = CachedResponse.objects().only("kind", "http_status", "error_code")
    return [
        {
            "kind": str(kind),
            "http_status": int(http_status),
            "error_code": error_code,
            "count": entries(kind=kind, http_status=http_status, error_code=error_code).count(),
        }
        for kind in sorted(entries.distinct("kind"))
        for http_status in sorted(entries(kind=kind).distinct("http_status"))
        for error_code in entries(kind=kind, http_status=http_status).distinct("error_code")
    ]


# /cache-reports/... endpoints


class ResponseReport(TypedDict):
    kind: str
    dataset: str
    config: Optional[str]
    split: Optional[str]
    http_status: int
    error_code: Optional[str]
    worker_version: Optional[str]
    dataset_git_revision: Optional[str]


class CacheReport(TypedDict):
    cache_reports: List[ResponseReport]
    next_cursor: str


class InvalidCursor(Exception):
    pass


class InvalidLimit(Exception):
    pass


def get_cache_reports(kind: str, cursor: Optional[str], limit: int) -> CacheReport:
    """
    Get a list of reports of the cache entries, along with the next cursor.
    See https://solovyov.net/blog/2020/api-pagination-design/.

    The "reports" are the cached entries, without the "content", "details" and "updated_at" fields.

    Args:
        kind (str): the kind of the cache entries
        cursor (`str`):
            An opaque string value representing a pointer to a specific FirstRowsResponse item in the dataset. The
            server returns results after the given pointer.
            An empty string means to start from the beginning.
        limit (strictly positive `int`):
            The maximum number of results.
    Returns:
        [`CacheReport`]: A dict with the list of reports and the next cursor. The next cursor is
        an empty string if there are no more items to be fetched.
    <Tip>
    Raises the following errors:
        - [`~libcommon.simple_cache.InvalidCursor`]
          If the cursor is invalid.
        - [`~libcommon.simple_cache.InvalidLimit`]
          If the limit is an invalid number.
    </Tip>
    """
    if not cursor:
        queryset = CachedResponse.objects(kind=kind)
    else:
        try:
            queryset = CachedResponse.objects(kind=kind, id__gt=ObjectId(cursor))
        except InvalidId as err:
            raise InvalidCursor("Invalid cursor.") from err
    if limit <= 0:
        raise InvalidLimit("Invalid limit.")
    objects = list(
        queryset.order_by("+id")
        .only(
            "id",
            "kind",
            "dataset",
            "config",
            "split",
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
                "kind": kind,
                "dataset": object.dataset,
                "config": object.config,
                "split": object.split,
                "http_status": object.http_status.value,
                "error_code": object.error_code,
                "worker_version": object.worker_version,
                "dataset_git_revision": object.dataset_git_revision,
            }
            for object in objects
        ],
        "next_cursor": "" if len(objects) < limit else str(objects[-1].id),
    }


# only for the tests
def _clean_cache_database() -> None:
    CachedResponse.drop_collection()  # type: ignore


# explicit re-export
__all__ = ["DoesNotExist"]
