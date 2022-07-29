import logging
import types
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Dict, Generic, List, Optional, Tuple, Type, TypedDict, TypeVar, Union

from mongoengine import Document, DoesNotExist, connect
from mongoengine.fields import (
    BooleanField,
    DateTimeField,
    DictField,
    EnumField,
    StringField,
)
from mongoengine.queryset.queryset import QuerySet

# START monkey patching ### hack ###
# see https://github.com/sbdchd/mongo-types#install
U = TypeVar("U", bound=Document)


def no_op(self, x):  # type: ignore
    return self


QuerySet.__class_getitem__ = types.MethodType(no_op, QuerySet)


logger = logging.getLogger(__name__)


class QuerySetManager(Generic[U]):
    def __get__(self, instance: object, cls: Type[U]) -> QuerySet[U]:
        return QuerySet(cls, cls._get_collection())


# END monkey patching ### hack ###


def connect_to_cache(database, host) -> None:
    connect(database, alias="cache", host=host)


def get_datetime() -> datetime:
    return datetime.now(timezone.utc)


# cache of the /splits endpoint
class SplitsResponse(Document):
    dataset_name = StringField(required=True, unique=True)
    http_status = EnumField(HTTPStatus, required=True)
    error_code = StringField(required=False)
    response = DictField(required=True)  # can be an error or a valid content. Not important here.
    details = DictField(required=False)  # can be a detailed error when we don't want to put it in the response.
    stale = BooleanField(required=False, default=False)
    updated_at = DateTimeField(default=get_datetime)

    meta = {
        "collection": "splitsResponse",
        "db_alias": "cache",
        "indexes": ["dataset_name", "http_status", "stale", "error_code"],
    }
    objects = QuerySetManager["SplitsResponse"]()


# cache of the /first-rows endpoint
class FirstRowsResponse(Document):
    dataset_name = StringField(required=True, unique_with=["config_name", "split_name"])
    config_name = StringField(required=True)
    split_name = StringField(required=True)
    http_status = EnumField(HTTPStatus, required=True)
    error_code = StringField(required=False)
    response = DictField(required=True)  # can be an error or a valid content. Not important here.
    details = DictField(required=False)  # can be a detailed error when we don't want to put it in the response.
    stale = BooleanField(required=False, default=False)
    updated_at = DateTimeField(default=get_datetime)

    meta = {
        "collection": "firstRowsResponse",
        "db_alias": "cache",
        "indexes": [
            ("dataset_name", "config_name", "split_name"),
            ("dataset_name", "http_status"),
            ("http_status", "dataset_name"),
            # ^ this index (reversed) is used for the "distinct" command to get the names of the valid datasets
            "error_code",
        ],
    }
    objects = QuerySetManager["FirstRowsResponse"]()


AnyResponse = TypeVar("AnyResponse", SplitsResponse, FirstRowsResponse)

# TODO: add logger.debug for each operation?


# /splits endpoint
# Note: we let the exceptions throw (ie DocumentTooLarge): it's the responsibility of the caller to manage them
def upsert_splits_response(
    dataset_name: str,
    response: Dict,
    http_status: HTTPStatus,
    error_code: Optional[str] = None,
    details: Optional[Dict] = None,
) -> None:
    SplitsResponse.objects(dataset_name=dataset_name).upsert_one(
        http_status=http_status,
        error_code=error_code,
        response=response,
        stale=False,
        details=details,
        updated_at=get_datetime(),
    )


def delete_splits_responses(dataset_name: str):
    SplitsResponse.objects(dataset_name=dataset_name).delete()


def mark_splits_responses_as_stale(dataset_name: str):
    SplitsResponse.objects(dataset_name=dataset_name).update(stale=True, updated_at=get_datetime())


# Note: we let the exceptions throw (ie DoesNotExist): it's the responsibility of the caller to manage them
def get_splits_response(dataset_name: str) -> Tuple[Dict, HTTPStatus, Optional[str]]:
    split_response = SplitsResponse.objects(dataset_name=dataset_name).get()
    return split_response.response, split_response.http_status, split_response.error_code


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
) -> None:
    FirstRowsResponse.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).upsert_one(
        http_status=http_status,
        error_code=error_code,
        response=response,
        stale=False,
        details=details,
        updated_at=get_datetime(),
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


# Note: we let the exceptions throw (ie DoesNotExist): it's the responsibility of the caller to manage them
def get_first_rows_response(
    dataset_name: str, config_name: str, split_name: str
) -> Tuple[Dict, HTTPStatus, Optional[str]]:
    first_rows_response = FirstRowsResponse.objects(
        dataset_name=dataset_name, config_name=config_name, split_name=split_name
    ).get()
    return first_rows_response.response, first_rows_response.http_status, first_rows_response.error_code


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


# admin /metrics endpoint


CountByHTTPStatus = Dict[str, int]


def get_entries_count_by_status(entries: QuerySet[AnyResponse]) -> CountByHTTPStatus:
    # return {http_status.name: entries(http_status=http_status).count() for http_status in HTTPStatus}
    return {
        HTTPStatus(http_status).name: entries(http_status=http_status).count()
        for http_status in sorted(entries.distinct("http_status"))
    }


def get_splits_responses_count_by_status() -> CountByHTTPStatus:
    return get_entries_count_by_status(SplitsResponse.objects)


def get_first_rows_responses_count_by_status() -> CountByHTTPStatus:
    return get_entries_count_by_status(FirstRowsResponse.objects)


CountByErrorCode = Dict[str, int]


def get_entries_count_by_error_code(entries: QuerySet[AnyResponse]) -> CountByErrorCode:
    return {error_code: entries(error_code=error_code).count() for error_code in entries.distinct("error_code")}


def get_splits_responses_count_by_error_code() -> CountByErrorCode:
    return get_entries_count_by_error_code(SplitsResponse.objects)


def get_first_rows_responses_count_by_error_code() -> CountByErrorCode:
    return get_entries_count_by_error_code(FirstRowsResponse.objects)


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


# /cache-reports endpoints


class _ErrorReport(TypedDict):
    message: str


class ErrorReport(_ErrorReport, total=False):
    cause_exception: str


class SplitsResponseReport(TypedDict):
    dataset: str
    status: int
    error: Optional[ErrorReport]


class FirstRowsResponseReport(TypedDict):
    dataset: str
    config: str
    split: str
    status: int
    error: Optional[ErrorReport]


def get_error(object: Union[SplitsResponse, FirstRowsResponse]) -> Optional[ErrorReport]:
    if object.http_status == HTTPStatus.OK:
        return None
    if "error" not in object.response:
        raise ValueError("Missing message in error response")
    report: ErrorReport = {"message": object.response["error"]}
    if "cause_exception" in object.response:
        report["cause_exception"] = object.response["cause_exception"]
    return report


def get_splits_response_reports() -> List[SplitsResponseReport]:
    return [
        {
            "dataset": response.dataset_name,
            "status": response.http_status.value,
            "error": get_error(response),
        }
        for response in SplitsResponse.objects()
    ]


def get_first_rows_response_reports() -> List[FirstRowsResponseReport]:
    return [
        {
            "dataset": response.dataset_name,
            "config": response.config_name,
            "split": response.split_name,
            "status": response.http_status.value,
            "error": get_error(response),
        }
        for response in FirstRowsResponse.objects()
    ]


# only for the tests
def _clean_database() -> None:
    SplitsResponse.drop_collection()  # type: ignore
    FirstRowsResponse.drop_collection()  # type: ignore


# explicit re-export
__all__ = ["DoesNotExist"]
