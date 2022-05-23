import enum
import logging
import types
from datetime import datetime
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

from libutils.exceptions import Status400Error, Status500Error, StatusError
from libutils.types import Split, SplitFullName
from mongoengine import Document, DoesNotExist, connect
from mongoengine.fields import (
    DateTimeField,
    DictField,
    EnumField,
    IntField,
    ListField,
    StringField,
)
from mongoengine.queryset.queryset import QuerySet
from pymongo.errors import DocumentTooLarge

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


class Status(enum.Enum):
    EMPTY = "empty"
    VALID = "valid"
    ERROR = "error"
    STALLED = "stalled"


# the purpose of this collection is to check if the dataset exists, which is its status and since when
class DbDataset(Document):
    dataset_name = StringField(required=True, unique=True)
    status = EnumField(Status, default=Status.EMPTY)
    since = DateTimeField(default=datetime.utcnow)

    meta = {"collection": "datasets", "db_alias": "cache", "indexes": ["dataset_name", "status"]}
    objects = QuerySetManager["DbDataset"]()


class SplitItem(TypedDict):
    dataset: str
    config: str
    split: str
    num_bytes: Optional[int]
    num_examples: Optional[int]


class SplitsResponse(TypedDict):
    splits: List[SplitItem]


def get_empty_rows_response() -> Dict[str, Any]:
    return {"columns": [], "rows": []}


class DbSplit(Document):
    dataset_name = StringField(required=True, unique_with=["config_name", "split_name"])
    config_name = StringField(required=True)
    split_name = StringField(required=True)
    split_idx = IntField(required=True, min_value=0)  # used to maintain the order
    num_bytes = IntField(min_value=0)
    num_examples = IntField(min_value=0)
    rows_response = DictField(required=True)
    status = EnumField(Status, default=Status.EMPTY)
    since = DateTimeField(default=datetime.utcnow)

    def to_split_item(self) -> SplitItem:
        return {
            "dataset": self.dataset_name,
            "config": self.config_name,
            "split": self.split_name,
            "num_bytes": self.num_bytes,
            "num_examples": self.num_examples,
        }

    def to_split_full_name(self) -> SplitFullName:
        return {"dataset_name": self.dataset_name, "config_name": self.config_name, "split_name": self.split_name}

    meta = {
        "collection": "splits",
        "db_alias": "cache",
        "indexes": ["status", ("dataset_name", "config_name", "split_name"), ("dataset_name", "status")],
    }
    objects = QuerySetManager["DbSplit"]()


class _BaseErrorItem(TypedDict):
    status_code: int
    exception: str
    message: str


class ErrorItem(_BaseErrorItem, total=False):
    # https://www.python.org/dev/peps/pep-0655/#motivation
    cause_exception: str
    cause_message: str
    cause_traceback: List[str]


class DbDatasetError(Document):
    dataset_name = StringField(required=True, unique=True)
    status_code = IntField(required=True)  # TODO: an enum
    exception = StringField(required=True)
    message = StringField(required=True)
    cause_exception = StringField()
    cause_message = StringField()
    cause_traceback = ListField(StringField())

    def to_item(self) -> ErrorItem:
        error: ErrorItem = {"status_code": self.status_code, "exception": self.exception, "message": self.message}
        if self.cause_exception and self.cause_message:
            error["cause_exception"] = self.cause_exception
            error["cause_message"] = self.cause_message
        if self.cause_traceback:
            error["cause_traceback"] = self.cause_traceback
        return error

    meta = {"collection": "dataset_errors", "db_alias": "cache"}
    objects = QuerySetManager["DbDatasetError"]()


class DbSplitError(Document):
    dataset_name = StringField(required=True, unique_with=["config_name", "split_name"])
    config_name = StringField(required=True)
    split_name = StringField(required=True)
    status_code = IntField(required=True)  # TODO: an enum
    exception = StringField(required=True)
    message = StringField(required=True)
    cause_exception = StringField()
    cause_message = StringField()
    cause_traceback = ListField(StringField())

    def to_item(self) -> ErrorItem:
        error: ErrorItem = {"status_code": self.status_code, "exception": self.exception, "message": self.message}
        if self.cause_exception and self.cause_message:
            error["cause_exception"] = self.cause_exception
            error["cause_message"] = self.cause_message
        if self.cause_traceback:
            error["cause_traceback"] = self.cause_traceback
        return error

    meta = {
        "collection": "split_errors",
        "db_alias": "cache",
        "indexes": [("dataset_name", "config_name", "split_name")],
    }
    objects = QuerySetManager["DbSplitError"]()


def upsert_dataset_error(dataset_name: str, error: StatusError) -> None:
    DbSplit.objects(dataset_name=dataset_name).delete()
    DbDataset.objects(dataset_name=dataset_name).upsert_one(status=Status.ERROR)
    DbDatasetError.objects(dataset_name=dataset_name).upsert_one(
        status_code=error.status_code,
        exception=error.exception,
        message=error.message,
        cause_exception=error.cause_exception,
        cause_message=error.cause_message,
        cause_traceback=error.cause_traceback,
    )


def upsert_dataset(dataset_name: str, new_split_full_names: List[SplitFullName]) -> None:
    DbDataset.objects(dataset_name=dataset_name).upsert_one(status=Status.VALID)
    DbDatasetError.objects(dataset_name=dataset_name).delete()

    split_full_names_to_delete = [
        o.to_split_full_name()
        for o in DbSplit.objects(dataset_name=dataset_name)
        if o.to_split_full_name() not in new_split_full_names
    ]

    for split_full_name in split_full_names_to_delete:
        delete_split(split_full_name)

    for split_idx, split_full_name in enumerate(new_split_full_names):
        create_or_mark_split_as_stalled(split_full_name, split_idx)


def upsert_split_error(dataset_name: str, config_name: str, split_name: str, error: StatusError) -> None:
    DbSplit.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).upsert_one(
        status=Status.ERROR
    )
    DbSplitError.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).upsert_one(
        status_code=error.status_code,
        exception=error.exception,
        message=error.message,
        cause_exception=error.cause_exception,
        cause_message=error.cause_message,
        cause_traceback=error.cause_traceback,
    )


def upsert_split(
    dataset_name: str,
    config_name: str,
    split_name: str,
    split: Split,
) -> None:
    try:
        DbSplit.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).upsert_one(
            status=Status.VALID,
            num_bytes=split["num_bytes"],
            num_examples=split["num_examples"],
            rows_response=split["rows_response"],  # TODO: a class method
        )
        DbSplitError.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).delete()
    except DocumentTooLarge as err:
        upsert_split_error(
            dataset_name, config_name, split_name, Status500Error("could not store the rows/ cache entry.", err)
        )


def delete_dataset_cache(dataset_name: str) -> None:
    DbDataset.objects(dataset_name=dataset_name).delete()
    DbSplit.objects(dataset_name=dataset_name).delete()
    DbDatasetError.objects(dataset_name=dataset_name).delete()
    DbSplitError.objects(dataset_name=dataset_name).delete()


def clean_database() -> None:
    DbDataset.drop_collection()  # type: ignore
    DbSplit.drop_collection()  # type: ignore
    DbDatasetError.drop_collection()  # type: ignore
    DbSplitError.drop_collection()  # type: ignore


def delete_split(split_full_name: SplitFullName):
    dataset_name = split_full_name["dataset_name"]
    config_name = split_full_name["config_name"]
    split_name = split_full_name["split_name"]
    DbSplit.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).delete()
    DbSplitError.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).delete()
    logger.debug(f"dataset '{dataset_name}': deleted split {split_name} from config {config_name}")


def create_empty_split(split_full_name: SplitFullName, split_idx: int):
    dataset_name = split_full_name["dataset_name"]
    config_name = split_full_name["config_name"]
    split_name = split_full_name["split_name"]
    DbSplit(
        dataset_name=dataset_name,
        config_name=config_name,
        split_name=split_name,
        status=Status.EMPTY,
        split_idx=split_idx,
        rows_response=get_empty_rows_response(),
    ).save()
    logger.debug(f"dataset '{dataset_name}': created empty split {split_name} in config {config_name}")


def create_empty_dataset(dataset_name: str):
    DbDataset(dataset_name=dataset_name).save()
    logger.debug(f"created empty dataset '{dataset_name}'")


def create_or_mark_dataset_as_stalled(dataset_name: str):
    try:
        DbDataset.objects(dataset_name=dataset_name).get()
        mark_dataset_as_stalled(dataset_name)
    except DoesNotExist:
        create_empty_dataset(dataset_name)


def mark_dataset_as_stalled(dataset_name: str):
    DbDataset.objects(dataset_name=dataset_name).update(status=Status.STALLED)
    logger.debug(f"marked dataset '{dataset_name}' as stalled")


def create_or_mark_split_as_stalled(split_full_name: SplitFullName, split_idx: int):
    try:
        DbSplit.objects(
            dataset_name=split_full_name["dataset_name"],
            config_name=split_full_name["config_name"],
            split_name=split_full_name["split_name"],
        ).get()
        mark_split_as_stalled(split_full_name, split_idx)
    except DoesNotExist:
        create_empty_split(split_full_name, split_idx)


def mark_split_as_stalled(split_full_name: SplitFullName, split_idx: int):
    dataset_name = split_full_name["dataset_name"]
    config_name = split_full_name["config_name"]
    split_name = split_full_name["split_name"]
    DbSplit.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).update(
        status=Status.STALLED, split_idx=split_idx
    )
    logger.debug(f"dataset '{dataset_name}': marked split {split_name} in config {config_name} as stalled")


def list_split_full_names_to_refresh(dataset_name: str):
    return [
        split.to_split_full_name()
        for split in DbSplit.objects(dataset_name=dataset_name, status__in=[Status.EMPTY, Status.STALLED])
    ]


# export


def should_dataset_be_refreshed(dataset_name: str) -> bool:
    try:
        dataset = DbDataset.objects(dataset_name=dataset_name).get()
        return dataset.status in [Status.STALLED, Status.EMPTY]
    except DoesNotExist:
        return True
    # ^ can also raise MultipleObjectsReturned, which should not occur -> we let the exception raise


def get_splits_response(dataset_name: str) -> Tuple[Union[SplitsResponse, None], Union[ErrorItem, None], int]:
    try:
        dataset = DbDataset.objects(dataset_name=dataset_name).get()
    except DoesNotExist as e:
        raise Status400Error("The dataset does not exist.") from e

    # ^ can also raise MultipleObjectsReturned, which should not occur -> we let the exception raise

    if dataset.status == Status.EMPTY:
        raise Status400Error("The dataset cache is empty.")
    if dataset.status == Status.ERROR:
        dataset_error = DbDatasetError.objects(dataset_name=dataset_name).get()
        # ^ can raise DoesNotExist or MultipleObjectsReturned, which should not occur -> we let the exception raise
        return None, dataset_error.to_item(), dataset_error.status_code

    splits_response: SplitsResponse = {
        "splits": [
            split.to_split_item() for split in DbSplit.objects(dataset_name=dataset_name).order_by("+split_idx")
        ]
    }
    return splits_response, None, 200


def get_rows_response(
    dataset_name: str,
    config_name: str,
    split_name: str,
) -> Tuple[Union[Dict[str, Any], None], Union[ErrorItem, None], int]:
    try:
        split = DbSplit.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).get()
    except DoesNotExist as e:
        raise Status400Error("The split does not exist.", e) from e

    # ^ can also raise MultipleObjectsReturned, which should not occur -> we let the exception raise

    if split.status == Status.EMPTY:
        raise Status400Error("The split cache is empty.")
        # ^ should not occur with the current logic
    if split.status == Status.ERROR:
        split_error = DbSplitError.objects(
            dataset_name=dataset_name, config_name=config_name, split_name=split_name
        ).get()
        # ^ can raise DoesNotExist or MultipleObjectsReturned, which should not occur -> we let the exception raise
        return None, split_error.to_item(), split_error.status_code

    return split.rows_response, None, 200


# special reports


def is_dataset_valid_or_stalled(dataset: DbDataset) -> bool:
    if dataset.status not in [Status.VALID, Status.STALLED]:
        return False

    splits = DbSplit.objects(dataset_name=dataset.dataset_name).only("status")
    return any(split.status in [Status.VALID, Status.STALLED] for split in splits)


def is_dataset_name_valid_or_stalled(dataset_name: str) -> bool:
    try:
        dataset = DbDataset.objects(dataset_name=dataset_name).get()
        return is_dataset_valid_or_stalled(dataset)
    except DoesNotExist:
        return False
    # ^ can also raise MultipleObjectsReturned, which should not occur -> we let the exception raise


class CountByCacheStatus(TypedDict):
    valid: int
    error: int
    missing: int


def get_dataset_cache_status(dataset_name: str) -> str:
    try:
        dataset = DbDataset.objects(dataset_name=dataset_name).get()
        if dataset.status == Status.EMPTY:
            return "missing"
        if dataset.status == Status.ERROR:
            return "error"
        splits = DbSplit.objects(dataset_name=dataset.dataset_name).only("status")
        if any(split.status == Status.EMPTY for split in splits):
            return "missing"
        elif any(split.status == Status.ERROR for split in splits):
            return "error"
        return "valid"
    except DoesNotExist:
        return "missing"
    # ^ can also raise MultipleObjectsReturned, which should not occur -> we let the exception raise


def get_datasets_count_by_cache_status(dataset_names: List[str]) -> CountByCacheStatus:
    dataset_statuses = [get_dataset_cache_status(x) for x in dataset_names]
    return {
        "valid": len([x for x in dataset_statuses if x == "valid"]),
        "error": len([x for x in dataset_statuses if x == "error"]),
        "missing": len([x for x in dataset_statuses if x == "missing"]),
    }


def get_valid_or_stalled_dataset_names() -> List[str]:
    # a dataset is considered valid if:
    # - the dataset is valid or stalled
    candidate_dataset_names = {
        d.dataset_name for d in DbDataset.objects(status__in=[Status.VALID, Status.STALLED]).only("dataset_name")
    }
    # - at least one of its splits is valid or stalled
    candidate_dataset_names_in_splits = {
        d.dataset_name for d in DbSplit.objects(status__in=[Status.VALID, Status.STALLED]).only("dataset_name")
    }
    candidate_dataset_names.intersection_update(candidate_dataset_names_in_splits)
    # note that the list is sorted alphabetically for consistency
    return sorted(candidate_dataset_names)


def get_dataset_names_with_status(status: str) -> List[str]:
    # TODO: take the splits statuses into account?
    return [d.dataset_name for d in DbDataset.objects(status=status).only("dataset_name")]


def get_datasets_count_with_status(status: Status) -> int:
    # TODO: take the splits statuses into account?
    return DbDataset.objects(status=status).count()


def get_splits_count_with_status(status: Status) -> int:
    # TODO: take the splits statuses into account?
    return DbSplit.objects(status=status).count()


class CountByStatus(TypedDict):
    empty: int
    error: int
    stalled: int
    valid: int


def get_datasets_count_by_status() -> CountByStatus:
    return {
        "empty": get_datasets_count_with_status(Status.EMPTY),
        "error": get_datasets_count_with_status(Status.ERROR),
        "stalled": get_datasets_count_with_status(Status.STALLED),
        "valid": get_datasets_count_with_status(Status.VALID),
    }


def get_splits_count_by_status() -> CountByStatus:
    return {
        "empty": get_splits_count_with_status(Status.EMPTY),
        "error": get_splits_count_with_status(Status.ERROR),
        "stalled": get_splits_count_with_status(Status.STALLED),
        "valid": get_splits_count_with_status(Status.VALID),
    }


class DatasetCacheReport(TypedDict):
    dataset: str
    status: str
    error: Union[Any, None]


def get_datasets_reports_with_error() -> List[DatasetCacheReport]:
    return [
        {"dataset": error.dataset_name, "status": Status.ERROR.value, "error": error.to_item()}
        for error in DbDatasetError.objects()
    ]


def get_datasets_reports_with_status(status: Status) -> List[DatasetCacheReport]:
    return [
        {"dataset": d.dataset_name, "status": status.value, "error": None}
        for d in DbDataset.objects(status=status).only("dataset_name")
    ]


class DatasetCacheReportsByStatus(TypedDict):
    empty: List[DatasetCacheReport]
    error: List[DatasetCacheReport]
    stalled: List[DatasetCacheReport]
    valid: List[DatasetCacheReport]


def get_datasets_reports_by_status() -> DatasetCacheReportsByStatus:
    # TODO: take the splits statuses into account?
    return {
        "empty": get_datasets_reports_with_status(Status.EMPTY),
        "error": get_datasets_reports_with_error(),
        "stalled": get_datasets_reports_with_status(Status.STALLED),
        "valid": get_datasets_reports_with_status(Status.VALID),
    }


class SplitCacheReport(TypedDict):
    dataset: str
    config: str
    split: str
    status: str
    error: Union[Any, None]


def get_splits_reports_with_error() -> List[SplitCacheReport]:
    return [
        {
            "dataset": error.dataset_name,
            "config": error.config_name,
            "split": error.split_name,
            "status": Status.ERROR.value,
            "error": error.to_item(),
        }
        for error in DbSplitError.objects()
    ]


def get_splits_reports_with_status(status: Status) -> List[SplitCacheReport]:
    return [
        {
            "dataset": d.dataset_name,
            "config": d.config_name,
            "split": d.split_name,
            "status": status.value,
            "error": None,
        }
        for d in DbSplit.objects(status=status).only("dataset_name", "config_name", "split_name")
    ]


class SplitCacheReportsByStatus(TypedDict):
    empty: List[SplitCacheReport]
    error: List[SplitCacheReport]
    stalled: List[SplitCacheReport]
    valid: List[SplitCacheReport]


def get_splits_reports_by_status() -> SplitCacheReportsByStatus:
    return {
        "empty": list(get_splits_reports_with_status(Status.EMPTY)),
        "error": get_splits_reports_with_error(),
        "stalled": get_splits_reports_with_status(Status.STALLED),
        "valid": get_splits_reports_with_status(Status.VALID),
    }
