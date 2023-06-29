# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import types
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Type,
    TypedDict,
    TypeVar,
)

import pandas as pd
from bson import ObjectId
from bson.errors import InvalidId
from mongoengine import Document
from mongoengine.errors import DoesNotExist
from mongoengine.fields import (
    DateTimeField,
    DictField,
    EnumField,
    FloatField,
    IntField,
    ObjectIdField,
    StringField,
)
from mongoengine.queryset.queryset import QuerySet

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.utils import JobParams, get_datetime

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


class SplitFullName(NamedTuple):
    """A split full name is a tuple of (dataset, config, split)."""

    dataset: str
    config: Optional[str]
    split: Optional[str]


# cache of any job
class CachedResponse(Document):
    """A response computed for a job, cached in the mongoDB database

    Args:
        kind (`str`): The kind of the cached response, identifies the job type
        dataset (`str`): The requested dataset.
        config (`str`, optional): The requested config, if any.
        split (`str`, optional): The requested split, if any.
        http_status (`HTTPStatus`): The HTTP status code.
        error_code (`str`, optional): The error code, if any.
        content (`dict`): The content of the cached response. Can be an error or a valid content.
        details (`dict`, optional): Additional details, eg. a detailed error that we don't want to send as a response.
        updated_at (`datetime`): When the cache entry has been last updated.
        job_runner_version (`int`): The version of the job runner that cached the response.
        dataset_git_revision (`str`): The commit (of the git dataset repo) used to generate the response.
        progress (`float`): Progress percentage (between 0. and 1.) if the result is not complete yet.
    """

    id = ObjectIdField(db_field="_id", primary_key=True, default=ObjectId)

    kind = StringField(required=True, unique_with=["dataset", "config", "split"])
    dataset = StringField(required=True)
    config = StringField()
    split = StringField()

    http_status = EnumField(HTTPStatus, required=True)
    error_code = StringField()
    content = DictField(required=True)
    dataset_git_revision = StringField()
    progress = FloatField(min_value=0.0, max_value=1.0)
    job_runner_version = IntField()

    details = DictField()
    updated_at = DateTimeField(default=get_datetime)

    meta = {
        "collection": CACHE_COLLECTION_RESPONSES,
        "db_alias": CACHE_MONGOENGINE_ALIAS,
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


class CacheEntryDoesNotExistError(DoesNotExist):
    pass


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
    job_runner_version: Optional[int] = None,
    dataset_git_revision: Optional[str] = None,
    progress: Optional[float] = None,
    updated_at: Optional[datetime] = None,
) -> None:
    CachedResponse.objects(kind=kind, dataset=dataset, config=config, split=split).upsert_one(
        content=content,
        http_status=http_status,
        error_code=error_code,
        details=details,
        dataset_git_revision=dataset_git_revision,
        progress=progress,
        updated_at=updated_at or get_datetime(),
        job_runner_version=job_runner_version,
    )


def upsert_response_params(
    kind: str,
    job_params: JobParams,
    content: Mapping[str, Any],
    http_status: HTTPStatus,
    error_code: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
    job_runner_version: Optional[int] = None,
    progress: Optional[float] = None,
    updated_at: Optional[datetime] = None,
) -> None:
    upsert_response(
        kind=kind,
        dataset=job_params["dataset"],
        config=job_params["config"],
        split=job_params["split"],
        content=content,
        dataset_git_revision=job_params["revision"],
        details=details,
        error_code=error_code,
        http_status=http_status,
        job_runner_version=job_runner_version,
        progress=progress,
        updated_at=updated_at,
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
    dataset_git_revision: Optional[str]
    progress: Optional[float]
    job_runner_version: Optional[int]


# Note: we let the exceptions throw: it's the responsibility of the caller to manage them
def get_response_without_content(
    kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None
) -> CacheEntryWithoutContent:
    try:
        response = (
            CachedResponse.objects(kind=kind, dataset=dataset, config=config, split=split)
            .only("http_status", "error_code", "job_runner_version", "dataset_git_revision", "progress")
            .get()
        )
    except DoesNotExist as e:
        raise CacheEntryDoesNotExistError(f"Cache entry does not exist: {kind=} {dataset=} {config=} {split=}") from e
    return {
        "http_status": response.http_status,
        "error_code": response.error_code,
        "dataset_git_revision": response.dataset_git_revision,
        "job_runner_version": response.job_runner_version,
        "progress": response.progress,
    }


def get_response_without_content_params(kind: str, job_params: JobParams) -> CacheEntryWithoutContent:
    return get_response_without_content(
        kind=kind, dataset=job_params["dataset"], config=job_params["config"], split=job_params["split"]
    )


class CacheEntryMetadata(CacheEntryWithoutContent):
    updated_at: datetime


# Note: we let the exceptions throw: it's the responsibility of the caller to manage them
def get_response_metadata(
    kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None
) -> CacheEntryMetadata:
    try:
        response = (
            CachedResponse.objects(kind=kind, dataset=dataset, config=config, split=split)
            .only("http_status", "error_code", "job_runner_version", "dataset_git_revision", "progress", "updated_at")
            .get()
        )
    except DoesNotExist as e:
        raise CacheEntryDoesNotExistError(f"Cache entry does not exist: {kind=} {dataset=} {config=} {split=}") from e
    return {
        "http_status": response.http_status,
        "error_code": response.error_code,
        "dataset_git_revision": response.dataset_git_revision,
        "job_runner_version": response.job_runner_version,
        "progress": response.progress,
        "updated_at": response.updated_at,
    }


class CacheEntry(CacheEntryWithoutContent):
    content: Mapping[str, Any]


class CacheEntryWithDetails(CacheEntry):
    details: Mapping[str, str]


class CachedArtifactError(Exception):
    kind: str
    dataset: str
    config: Optional[str]
    split: Optional[str]
    cache_entry_with_details: CacheEntryWithDetails
    enhanced_details: Dict[str, Any]

    def __init__(
        self,
        message: str,
        kind: str,
        dataset: str,
        config: Optional[str],
        split: Optional[str],
        cache_entry_with_details: CacheEntryWithDetails,
    ):
        super().__init__(message)
        self.kind = kind
        self.dataset = dataset
        self.config = config
        self.split = split
        self.cache_entry_with_details = cache_entry_with_details
        self.enhanced_details: Dict[str, Any] = dict(self.cache_entry_with_details["details"].items())
        self.enhanced_details["copied_from_artifact"] = {
            "kind": self.kind,
            "dataset": self.dataset,
            "config": self.config,
            "split": self.split,
        }


# Note: we let the exceptions throw: it's the responsibility of the caller to manage them
def get_response(kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None) -> CacheEntry:
    try:
        response = (
            CachedResponse.objects(kind=kind, dataset=dataset, config=config, split=split)
            .only("content", "http_status", "error_code", "job_runner_version", "dataset_git_revision", "progress")
            .get()
        )
    except DoesNotExist as e:
        raise CacheEntryDoesNotExistError(f"Cache entry does not exist: {kind=} {dataset=} {config=} {split=}") from e
    return {
        "content": response.content,
        "http_status": response.http_status,
        "error_code": response.error_code,
        "job_runner_version": response.job_runner_version,
        "dataset_git_revision": response.dataset_git_revision,
        "progress": response.progress,
    }


# Note: we let the exceptions throw: it's the responsibility of the caller to manage them
def get_response_with_details(
    kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None
) -> CacheEntryWithDetails:
    try:
        response = (
            CachedResponse.objects(kind=kind, dataset=dataset, config=config, split=split)
            .only(
                "content",
                "http_status",
                "error_code",
                "job_runner_version",
                "dataset_git_revision",
                "progress",
                "details",
            )
            .get()
        )
    except DoesNotExist as e:
        raise CacheEntryDoesNotExistError(f"Cache entry does not exist: {kind=} {dataset=} {config=} {split=}") from e
    return {
        "content": response.content,
        "http_status": response.http_status,
        "error_code": response.error_code,
        "job_runner_version": response.job_runner_version,
        "dataset_git_revision": response.dataset_git_revision,
        "progress": response.progress,
        "details": response.details,
    }


CACHED_RESPONSE_NOT_FOUND = "CachedResponseNotFound"


def get_response_or_missing_error(
    kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None
) -> CacheEntryWithDetails:
    try:
        response = get_response_with_details(kind=kind, dataset=dataset, config=config, split=split)
    except CacheEntryDoesNotExistError:
        response = CacheEntryWithDetails(
            content={
                "error": (
                    f"Cached response not found for kind {kind}, dataset {dataset}, config {config}, split {split}"
                )
            },
            http_status=HTTPStatus.NOT_FOUND,
            error_code=CACHED_RESPONSE_NOT_FOUND,
            dataset_git_revision=None,
            job_runner_version=None,
            progress=None,
            details={},
        )
    return response


@dataclass
class BestResponse:
    kind: str
    response: CacheEntryWithDetails


def get_best_response(
    kinds: List[str], dataset: str, config: Optional[str] = None, split: Optional[str] = None
) -> BestResponse:
    """
    Get the best response from a list of cache kinds.

    Best means:
    - the first success response with the highest progress,
    - else: the first error response (including cache miss)

    Args:
        kinds (`List[str]`):
            A non-empty list of cache kinds to look responses for.
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.
        config (`str`, optional):
            A config name.
        split (`str`, optional):
            A split name.
    Returns:
        BestResponse: The best response (object with fields: kind and response). The response can be an error,
          including a cache miss (error code: `CachedResponseNotFound`)
    """
    if not kinds:
        raise ValueError("kinds must be a non-empty list")
    best_response_candidates = [
        BestResponse(
            kind=kind, response=get_response_or_missing_error(kind=kind, dataset=dataset, config=config, split=split)
        )
        for kind in kinds
    ]
    max_index = 0
    max_value = float("-inf")
    for index, candidate in enumerate(best_response_candidates):
        if candidate.response["http_status"] >= HTTPStatus.BAD_REQUEST.value:
            # only the first error response is considered
            continue
        value = (
            0.0
            if candidate.response["progress"] is None or candidate.response["progress"] < 0.0
            else candidate.response["progress"]
        )
        if value > max_value:
            max_value = value
            max_index = index
    return best_response_candidates[max_index]


def get_previous_step_or_raise(
    kinds: List[str], dataset: str, config: Optional[str] = None, split: Optional[str] = None
) -> BestResponse:
    """Get the previous step from the cache, or raise an exception if it failed."""
    best_response = get_best_response(kinds=kinds, dataset=dataset, config=config, split=split)
    if best_response.response["http_status"] != HTTPStatus.OK:
        raise CachedArtifactError(
            message="The previous step failed.",
            kind=best_response.kind,
            dataset=dataset,
            config=config,
            split=split,
            cache_entry_with_details=best_response.response,
        )
    return best_response


def get_valid_datasets(kind: str) -> Set[str]:
    return set(CachedResponse.objects(kind=kind, http_status=HTTPStatus.OK).distinct("dataset"))


def is_valid_for_kinds(dataset: str, kinds: List[str]) -> bool:
    return CachedResponse.objects(dataset=dataset, kind__in=kinds, http_status=HTTPStatus.OK).count() > 0


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


class CacheReport(TypedDict):
    kind: str
    dataset: str
    config: Optional[str]
    split: Optional[str]
    http_status: int
    error_code: Optional[str]
    details: Mapping[str, Any]
    updated_at: datetime
    job_runner_version: Optional[int]
    dataset_git_revision: Optional[str]
    progress: Optional[float]


class CacheReportsPage(TypedDict):
    cache_reports: List[CacheReport]
    next_cursor: str


class InvalidCursor(Exception):
    pass


class InvalidLimit(Exception):
    pass


def get_cache_reports(kind: str, cursor: Optional[str], limit: int) -> CacheReportsPage:
    """
    Get a list of reports of the cache entries, along with the next cursor.
    See https://solovyov.net/blog/2020/api-pagination-design/.

    The "reports" are the cached entries, without the "content", "details" and "updated_at" fields.

    Args:
        kind (str): the kind of the cache entries
        cursor (`str`):
            An opaque string value representing a pointer to a specific CachedResponse item in the dataset. The
            server returns results after the given pointer.
            An empty string means to start from the beginning.
        limit (strictly positive `int`):
            The maximum number of results.
    Returns:
        [`CacheReportsPage`]: A dict with the list of reports and the next cursor. The next cursor is
        an empty string if there are no more items to be fetched.
    Raises the following errors:
        - [`~simple_cache.InvalidCursor`]
          If the cursor is invalid.
        - [`~simple_cache.InvalidLimit`]
          If the limit is an invalid number.
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
    objects = list(queryset.order_by("+id").exclude("content").limit(limit))
    return {
        "cache_reports": [
            {
                "kind": kind,
                "dataset": object.dataset,
                "config": object.config,
                "split": object.split,
                "http_status": object.http_status.value,
                "error_code": object.error_code,
                "details": object.details,
                "updated_at": object.updated_at,
                "job_runner_version": object.job_runner_version,
                "dataset_git_revision": object.dataset_git_revision,
                "progress": object.progress,
            }
            for object in objects
        ],
        "next_cursor": "" if len(objects) < limit else str(objects[-1].id),
    }


def get_outdated_split_full_names_for_step(kind: str, current_version: int) -> List[SplitFullName]:
    responses = CachedResponse.objects(kind=kind, job_runner_version__lt=current_version).only(
        "dataset", "config", "split"
    )
    return [
        SplitFullName(dataset=response.dataset, config=response.config, split=response.split) for response in responses
    ]


def get_dataset_responses_without_content_for_kind(kind: str, dataset: str) -> List[CacheReport]:
    responses = CachedResponse.objects(kind=kind, dataset=dataset).exclude("content")
    return [
        {
            "kind": response.kind,
            "dataset": response.dataset,
            "config": response.config,
            "split": response.split,
            "http_status": response.http_status,
            "error_code": response.error_code,
            "details": response.details,
            "updated_at": response.updated_at,
            "job_runner_version": response.job_runner_version,
            "dataset_git_revision": response.dataset_git_revision,
            "progress": response.progress,
        }
        for response in responses
    ]


class CacheReportWithContent(CacheReport):
    content: Mapping[str, Any]


class CacheReportsWithContentPage(TypedDict):
    cache_reports_with_content: List[CacheReportWithContent]
    next_cursor: str


def get_cache_reports_with_content(kind: str, cursor: Optional[str], limit: int) -> CacheReportsWithContentPage:
    """
    Get a list of the cache report with content, along with the next cursor.
    See https://solovyov.net/blog/2020/api-pagination-design/.

    The cache reports contain all the fields of the object, including the "content" field.

    Args:
        kind (str): the kind of the cache entries
        cursor (`str`):
            An opaque string value representing a pointer to a specific CachedResponse item in the dataset. The
            server returns results after the given pointer.
            An empty string means to start from the beginning.
        limit (strictly positive `int`):
            The maximum number of results.
    Returns:
        [`CacheReportsWithContentPage`]: A dict with the list of reports and the next cursor. The next cursor is
        an empty string if there are no more items to be fetched.
    Raises the following errors:
        - [`~simple_cache.InvalidCursor`]
          If the cursor is invalid.
        - [`~simple_cache.InvalidLimit`]
          If the limit is an invalid number.
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
    objects = list(queryset.order_by("+id").limit(limit))
    return {
        "cache_reports_with_content": [
            {
                "kind": kind,
                "dataset": object.dataset,
                "config": object.config,
                "split": object.split,
                "http_status": object.http_status.value,
                "error_code": object.error_code,
                "content": object.content,
                "job_runner_version": object.job_runner_version,
                "dataset_git_revision": object.dataset_git_revision,
                "details": object.details,
                "updated_at": object.updated_at,
                "progress": object.progress,
            }
            for object in objects
        ],
        "next_cursor": "" if len(objects) < limit else str(objects[-1].id),
    }


class CacheEntryFullMetadata(CacheEntryMetadata):
    kind: str
    dataset: str
    config: Optional[str]
    split: Optional[str]


def _get_df(entries: List[CacheEntryFullMetadata]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kind": pd.Series([entry["kind"] for entry in entries], dtype="category"),
            "dataset": pd.Series([entry["dataset"] for entry in entries], dtype="str"),
            "config": pd.Series([entry["config"] for entry in entries], dtype="str"),
            "split": pd.Series([entry["split"] for entry in entries], dtype="str"),
            "http_status": pd.Series(
                [entry["http_status"] for entry in entries], dtype="category"
            ),  # check if it's working as expected
            "error_code": pd.Series([entry["error_code"] for entry in entries], dtype="category"),
            "dataset_git_revision": pd.Series([entry["dataset_git_revision"] for entry in entries], dtype="str"),
            "job_runner_version": pd.Series([entry["job_runner_version"] for entry in entries], dtype=pd.Int16Dtype()),
            "progress": pd.Series([entry["progress"] for entry in entries], dtype="float"),
            "updated_at": pd.Series(
                [entry["updated_at"] for entry in entries], dtype="datetime64[ns]"
            ),  # check if it's working as expected
        }
    )
    # ^ does not seem optimal at all, but I get the types right


def get_cache_entries_df(dataset: str, cache_kinds: Optional[List[str]] = None) -> pd.DataFrame:
    filters = {}
    if cache_kinds:
        filters["kind__in"] = cache_kinds
    return _get_df(
        [
            {
                "kind": response.kind,
                "dataset": response.dataset,
                "config": response.config,
                "split": response.split,
                "http_status": response.http_status,
                "error_code": response.error_code,
                "dataset_git_revision": response.dataset_git_revision,
                "job_runner_version": response.job_runner_version,
                "progress": response.progress,
                "updated_at": response.updated_at,
            }
            for response in CachedResponse.objects(dataset=dataset, **filters).only(
                "kind",
                "dataset",
                "config",
                "split",
                "http_status",
                "error_code",
                "job_runner_version",
                "dataset_git_revision",
                "progress",
                "updated_at",
            )
        ]
    )


def has_some_cache(dataset: str) -> bool:
    return CachedResponse.objects(dataset=dataset).count() > 0


def fetch_names(
    dataset: str, config: Optional[str], cache_kinds: List[str], names_field: str, name_field: str
) -> List[str]:
    """
    Fetch a list of names from the cache database.

    If no entry is found in cache, return an empty list. Exceptions are silently caught.

    Args:
        dataset (str): The dataset name.
        config (Optional[str]): The config name. Only needed for split names.
        cache_kinds (List[str]): The cache kinds to fetch, eg ["dataset-config-names"],
          or ["config-split-names-from-streaming", "config-split-names-from-info"].
        names_field (str): The name of the field containing the list of names, eg: "config_names", or "splits".
        name_field (str): The name of the field containing the name, eg: "config", or "split".

    Returns:
        List[str]: The list of names.
    """
    try:
        names = []
        best_response = get_best_response(kinds=cache_kinds, dataset=dataset, config=config)
        for name_item in best_response.response["content"][names_field]:
            name = name_item[name_field]
            if not isinstance(name, str):
                raise ValueError(f"Invalid name: {name}, type should be str, got: {type(name)}")
            names.append(name)
        return names
    except Exception:
        return []


# only for the tests
def _clean_cache_database() -> None:
    CachedResponse.drop_collection()  # type: ignore
