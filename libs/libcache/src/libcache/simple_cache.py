import enum
import logging
import types
from datetime import datetime, timezone
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


# subset of https://docs.python.org/3/library/http.html#http.HTTPStatus
class HTTPStatus(enum.Enum):
    OK = "200"
    BAD_REQUEST = "400"
    INTERNAL_SERVER_ERROR = "500"


def get_datetime() -> datetime:
    return datetime.now(timezone.utc)


# cache of the /splits endpoint
class SplitsResponse(Document):
    dataset_name = StringField(required=True, unique=True)
    http_status = EnumField(HTTPStatus, required=True)
    response = DictField(required=True)  # can be an error or a valid content. Not important here.
    details = DictField(required=False)  # can be a detailed error when we don't want to put it in the response.
    stale = BooleanField(required=False, default=False)
    updated_at = DateTimeField(default=get_datetime)

    meta = {
        "collection": "splitsResponse",
        "db_alias": "cache",
        "indexes": ["dataset_name", "http_status", "stale"],
    }
    objects = QuerySetManager["SplitsResponse"]()


# cache of the /first-rows endpoint
class FirstRowsResponse(Document):
    dataset_name = StringField(required=True, unique_with=["config_name", "split_name"])
    config_name = StringField(required=True)
    split_name = StringField(required=True)
    http_status = EnumField(HTTPStatus, required=True)
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
        ],
    }
    objects = QuerySetManager["FirstRowsResponse"]()


AnyResponse = TypeVar("AnyResponse", SplitsResponse, FirstRowsResponse)

# TODO: add logger.debug for each operation?


# /splits endpoint
# Note: we let the exceptions throw (ie DocumentTooLarge): it's the responsibility of the caller to manage them
def upsert_splits_response(
    dataset_name: str, response: Dict, http_status: HTTPStatus, details: Optional[Dict] = None
) -> None:
    SplitsResponse.objects(dataset_name=dataset_name).upsert_one(
        http_status=http_status,
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
def get_splits_response(dataset_name: str) -> Tuple[Dict, HTTPStatus]:
    split_response = SplitsResponse.objects(dataset_name=dataset_name).get()
    return split_response.response, split_response.http_status


# /first-rows endpoint
# Note: we let the exceptions throw (ie DocumentTooLarge): it's the responsibility of the caller to manage them
def upsert_first_rows_response(
    dataset_name: str,
    config_name: str,
    split_name: str,
    response: Dict,
    http_status: HTTPStatus,
    details: Optional[Dict] = None,
) -> None:
    FirstRowsResponse.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).upsert_one(
        http_status=http_status, response=response, stale=False, details=details, updated_at=get_datetime()
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
def get_first_rows_response(dataset_name: str, config_name: str, split_name: str) -> Tuple[Dict, HTTPStatus]:
    first_rows_response = FirstRowsResponse.objects(
        dataset_name=dataset_name, config_name=config_name, split_name=split_name
    ).get()
    return first_rows_response.response, first_rows_response.http_status


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


# /pending-jobs endpoint


class CountByHTTPStatus(TypedDict):
    OK: int
    BAD_REQUEST: int
    INTERNAL_SERVER_ERROR: int


def get_entries_count_by_status(entries: QuerySet[AnyResponse]) -> CountByHTTPStatus:
    # ensure that all the statuses are present, even if equal to zero
    # note: we repeat the values instead of looping on Status because we don't know how to get the types right in mypy
    # result: CountByStatus = {s.value: entries(status=s.value).count() for s in Status} # <- doesn't work in mypy
    # see https://stackoverflow.com/a/67292548/7351594
    return {
        "OK": entries(http_status=HTTPStatus.OK.value).count(),
        "BAD_REQUEST": entries(http_status=HTTPStatus.BAD_REQUEST.value).count(),
        "INTERNAL_SERVER_ERROR": entries(http_status=HTTPStatus.INTERNAL_SERVER_ERROR).count(),
    }


def get_splits_responses_count_by_status() -> CountByHTTPStatus:
    # TODO: take the splits statuses into account?
    return get_entries_count_by_status(SplitsResponse.objects)


def get_first_rows_responses_count_by_status() -> CountByHTTPStatus:
    return get_entries_count_by_status(FirstRowsResponse.objects)


# /cache-reports endpoints


class _ErrorReport(TypedDict):
    message: str


class ErrorReport(_ErrorReport, total=False):
    cause_exception: str


class SplitsResponseReport(TypedDict):
    dataset: str
    status: str
    error: Optional[ErrorReport]


class FirstRowsResponseReport(TypedDict):
    dataset: str
    config: str
    split: str
    status: str
    error: Optional[ErrorReport]


def get_error(object: Union[SplitsResponse, FirstRowsResponse]) -> Optional[ErrorReport]:
    if object.http_status == HTTPStatus.OK:
        return None
    if "message" not in object.response:
        raise ValueError("Missing message in error response")
    report: ErrorReport = {"message": object.response["message"]}
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
