import enum
import logging
import sys
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

from datasets_preview_backend.config import MONGO_CACHE_DATABASE, MONGO_URL
from datasets_preview_backend.constants import DEFAULT_MIN_CELL_BYTES
from datasets_preview_backend.exceptions import (
    Status400Error,
    Status500Error,
    StatusError,
)
from datasets_preview_backend.models.column import (
    ClassLabelColumn,
    ColumnDict,
    ColumnType,
)
from datasets_preview_backend.models.dataset import (
    SplitFullName,
    get_dataset_split_full_names,
)
from datasets_preview_backend.models.split import Split, get_split
from datasets_preview_backend.utils import orjson_dumps

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


def connect_to_cache() -> None:
    connect(MONGO_CACHE_DATABASE, alias="cache", host=MONGO_URL)


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

    meta = {"collection": "datasets", "db_alias": "cache"}
    objects = QuerySetManager["DbDataset"]()


class SplitItem(TypedDict):
    dataset: str
    config: str
    split: str
    num_bytes: Optional[int]
    num_examples: Optional[int]


class SplitsResponse(TypedDict):
    splits: List[SplitItem]


class DbSplit(Document):
    dataset_name = StringField(required=True, unique_with=["config_name", "split_name"])
    config_name = StringField(required=True)
    split_name = StringField(required=True)
    split_idx = IntField(required=True, min_value=0)  # used to maintain the order
    num_bytes = IntField(min_value=0)
    num_examples = IntField(min_value=0)
    status = EnumField(Status, default=Status.EMPTY)
    since = DateTimeField(default=datetime.utcnow)

    def to_item(self) -> SplitItem:
        return {
            "dataset": self.dataset_name,
            "config": self.config_name,
            "split": self.split_name,
            "num_bytes": self.num_bytes,
            "num_examples": self.num_examples,
        }

    def to_split_full_name(self) -> SplitFullName:
        return {"dataset_name": self.dataset_name, "config_name": self.config_name, "split_name": self.split_name}

    meta = {"collection": "splits", "db_alias": "cache"}
    objects = QuerySetManager["DbSplit"]()


class RowItem(TypedDict):
    dataset: str
    config: str
    split: str
    row_idx: int
    row: Dict[str, Any]
    truncated_cells: List[str]


class DbRow(Document):
    dataset_name = StringField(required=True, unique_with=["config_name", "split_name", "row_idx"])
    config_name = StringField(required=True)
    split_name = StringField(required=True)
    row_idx = IntField(required=True, min_value=0)
    row = DictField(required=True)

    def to_item(self) -> RowItem:
        return {
            "dataset": self.dataset_name,
            "config": self.config_name,
            "split": self.split_name,
            "row_idx": self.row_idx,
            "row": self.row,
            "truncated_cells": [],
        }

    meta = {"collection": "rows", "db_alias": "cache"}
    objects = QuerySetManager["DbRow"]()


class ColumnItem(TypedDict):
    dataset: str
    config: str
    split: str
    column_idx: int
    column: ColumnDict


class RowsResponse(TypedDict):
    columns: List[ColumnItem]
    rows: List[RowItem]


class DbColumn(Document):
    dataset_name = StringField(required=True, unique_with=["config_name", "split_name", "name"])
    config_name = StringField(required=True)
    split_name = StringField(required=True)
    column_idx = IntField(required=True, min_value=0)
    name = StringField(required=True)
    type = EnumField(ColumnType, required=True)
    labels = ListField(StringField())

    def to_item(self) -> ColumnItem:
        column: ColumnDict = {"name": self.name, "type": self.type.name}
        if self.labels:
            column["labels"] = self.labels
        return {
            "dataset": self.dataset_name,
            "config": self.config_name,
            "split": self.split_name,
            "column_idx": self.column_idx,
            "column": column,
        }

    meta = {"collection": "columns", "db_alias": "cache"}
    objects = QuerySetManager["DbColumn"]()


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

    meta = {"collection": "split_errors", "db_alias": "cache"}
    objects = QuerySetManager["DbSplitError"]()


def upsert_dataset_error(dataset_name: str, error: StatusError) -> None:
    DbSplit.objects(dataset_name=dataset_name).delete()
    DbRow.objects(dataset_name=dataset_name).delete()
    DbColumn.objects(dataset_name=dataset_name).delete()
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

    current_split_full_names = [o.to_split_full_name() for o in DbSplit.objects(dataset_name=dataset_name)]

    for split_full_name in current_split_full_names:
        if split_full_name not in new_split_full_names:
            # delete the splits that disappeared
            delete_split(split_full_name)

    for split_idx, split_full_name in enumerate(new_split_full_names):
        if split_full_name not in current_split_full_names:
            # create the new empty splits
            create_split(split_full_name, split_idx)
        else:
            # mark all the existing splits as stalled
            mark_split_as_stalled(split_full_name, split_idx)


def upsert_split_error(dataset_name: str, config_name: str, split_name: str, error: StatusError) -> None:
    DbRow.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).delete()
    DbColumn.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).delete()
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


def upsert_split(dataset_name: str, config_name: str, split_name: str, split: Split) -> None:
    rows = split["rows"]
    columns = split["columns"]

    DbSplit.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).upsert_one(
        status=Status.VALID, num_bytes=split["num_bytes"], num_examples=split["num_examples"]
    )
    DbSplitError.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).delete()

    DbRow.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).delete()
    for row_idx, row in enumerate(rows):
        DbRow(
            dataset_name=dataset_name,
            config_name=config_name,
            split_name=split_name,
            row_idx=row_idx,
            row=row,
        ).save()

    DbColumn.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).delete()
    for column_idx, column in enumerate(columns):
        db_column = DbColumn(
            dataset_name=dataset_name,
            config_name=config_name,
            split_name=split_name,
            column_idx=column_idx,
            name=column.name,
            type=column.type,
        )
        # TODO: seems like suboptimal code, introducing unnecessary coupling
        if isinstance(column, ClassLabelColumn):
            db_column.labels = column.labels
        db_column.save()


def delete_dataset_cache(dataset_name: str) -> None:
    DbDataset.objects(dataset_name=dataset_name).delete()
    DbSplit.objects(dataset_name=dataset_name).delete()
    DbRow.objects(dataset_name=dataset_name).delete()
    DbColumn.objects(dataset_name=dataset_name).delete()
    DbDatasetError.objects(dataset_name=dataset_name).delete()


def clean_database() -> None:
    DbDataset.drop_collection()  # type: ignore
    DbSplit.drop_collection()  # type: ignore
    DbRow.drop_collection()  # type: ignore
    DbColumn.drop_collection()  # type: ignore
    DbDatasetError.drop_collection()  # type: ignore


def refresh_dataset_split_full_names(dataset_name: str, hf_token: Optional[str] = None) -> List[SplitFullName]:
    try:
        split_full_names = get_dataset_split_full_names(dataset_name, hf_token)
        upsert_dataset(dataset_name, split_full_names)
        logger.debug(f"dataset '{dataset_name}' is valid, cache updated")
        return split_full_names
    except StatusError as err:
        upsert_dataset_error(dataset_name, err)
        logger.debug(f"dataset '{dataset_name}' had error, cache updated")
        raise
    except Exception as err:
        upsert_dataset_error(dataset_name, Status500Error(str(err)))
        logger.debug(f"dataset '{dataset_name}' had error, cache updated")
        raise


def delete_split(split_full_name: SplitFullName):
    dataset_name = split_full_name["dataset_name"]
    config_name = split_full_name["config_name"]
    split_name = split_full_name["split_name"]
    DbRow.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).delete()
    DbColumn.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).delete()
    DbSplit.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).delete()
    # DbSplitError.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).delete()
    logger.debug(f"dataset '{dataset_name}': deleted split {split_name} from config {config_name}")


def create_split(split_full_name: SplitFullName, split_idx: int):
    dataset_name = split_full_name["dataset_name"]
    config_name = split_full_name["config_name"]
    split_name = split_full_name["split_name"]
    DbSplit(
        dataset_name=dataset_name,
        config_name=config_name,
        split_name=split_name,
        status=Status.EMPTY,
        split_idx=split_idx,
    ).save()
    logger.debug(f"dataset '{dataset_name}': created split {split_name} in config {config_name}")


def mark_split_as_stalled(split_full_name: SplitFullName, split_idx: int):
    dataset_name = split_full_name["dataset_name"]
    config_name = split_full_name["config_name"]
    split_name = split_full_name["split_name"]
    DbSplit.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).update(
        status=Status.STALLED, split_idx=split_idx
    )
    logger.debug(f"dataset '{dataset_name}': marked split {split_name} in config {config_name} as stalled")


def refresh_split(
    dataset_name: str,
    config_name: str,
    split_name: str,
    hf_token: Optional[str] = None,
    max_size_fallback: Optional[int] = None,
):
    try:
        split = get_split(
            dataset_name, config_name, split_name, hf_token=hf_token, max_size_fallback=max_size_fallback
        )
        upsert_split(dataset_name, config_name, split_name, split)
        logger.debug(
            f"split '{split_name}' from dataset '{dataset_name}' in config '{config_name}' is valid, cache updated"
        )
    except StatusError as err:
        upsert_split_error(dataset_name, config_name, split_name, err)
        logger.debug(
            f"split '{split_name}' from dataset '{dataset_name}' in config '{config_name}' had error, cache updated"
        )
        raise
    except Exception as err:
        upsert_split_error(dataset_name, config_name, split_name, Status500Error(str(err)))
        logger.debug(
            f"split '{split_name}' from dataset '{dataset_name}' in config '{config_name}' had error, cache updated"
        )
        raise


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
        raise Status400Error("Not found. Maybe the cache is missing, or maybe the dataset does not exist.") from e

    # ^ can also raise MultipleObjectsReturned, which should not occur -> we let the exception raise

    if dataset.status == Status.EMPTY:
        raise Status400Error("Not found. Cache is waiting to be refreshed.")
        # ^ should not occur with the current logic
    if dataset.status == Status.ERROR:
        dataset_error = DbDatasetError.objects(dataset_name=dataset_name).get()
        # ^ can raise DoesNotExist or MultipleObjectsReturned, which should not occur -> we let the exception raise
        return None, dataset_error.to_item(), dataset_error.status_code

    splits_response: SplitsResponse = {
        "splits": [split.to_item() for split in DbSplit.objects(dataset_name=dataset_name).order_by("+split_idx")]
    }
    return splits_response, None, 200


def get_size_in_bytes(obj: Any):
    return sys.getsizeof(orjson_dumps(obj))
    # ^^ every row is transformed here in a string, because it corresponds to
    # the size the row will contribute in the JSON response to /rows endpoint.
    # The size of the string is measured in bytes.
    # An alternative would have been to look at the memory consumption (pympler) but it's
    # less related to what matters here (size of the JSON, number of characters in the
    # dataset viewer table on the hub)


def truncate_cell(cell: Any, min_cell_bytes: int) -> str:
    return orjson_dumps(cell)[:min_cell_bytes].decode("utf8", "ignore")


# Mutates row_item, and returns it anyway
def truncate_row_item(row_item: RowItem) -> RowItem:
    min_cell_bytes = DEFAULT_MIN_CELL_BYTES
    row = {}
    for column_name, cell in row_item["row"].items():
        # for now: all the cells, but the smallest ones, are truncated
        cell_bytes = get_size_in_bytes(cell)
        if cell_bytes > min_cell_bytes:
            row_item["truncated_cells"].append(column_name)
            row[column_name] = truncate_cell(cell, min_cell_bytes)
        else:
            row[column_name] = cell
    row_item["row"] = row
    return row_item


# Mutates row_items, and returns them anyway
def truncate_row_items(row_items: List[RowItem], rows_max_bytes: int) -> List[RowItem]:
    # compute the current size
    rows_bytes = sum(get_size_in_bytes(row_item) for row_item in row_items)

    # Loop backwards, so that the last rows are truncated first
    for row_item in reversed(row_items):
        previous_size = get_size_in_bytes(row_item)
        row_item = truncate_row_item(row_item)
        new_size = get_size_in_bytes(row_item)
        rows_bytes += new_size - previous_size
        row_idx = row_item["row_idx"]
        logger.debug(f"the size of the rows is now ({rows_bytes}) after truncating row idx={row_idx}")
        if rows_bytes < rows_max_bytes:
            break
    return row_items


def to_row_items(
    rows: QuerySet[DbRow], rows_max_bytes: Optional[int], rows_min_number: Optional[int]
) -> List[RowItem]:
    row_items = []
    rows_bytes = 0
    if rows_min_number is None:
        rows_min_number = 0
    else:
        logger.debug(f"min number of rows in the response: '{rows_min_number}'")
    if rows_max_bytes is not None:
        logger.debug(f"max number of bytes in the response: '{rows_max_bytes}'")

    # two restrictions must be enforced:
    # - at least rows_min_number rows
    # - at most rows_max_bytes bytes
    # To enforce this:
    # 1. first get the first rows_min_number rows
    for row in rows[:rows_min_number]:
        row_item = row.to_item()
        if rows_max_bytes is not None:
            rows_bytes += get_size_in_bytes(row_item)
        row_items.append(row_item)

    # 2. if the total is over the bytes limit, truncate the values, iterating backwards starting
    # from the last rows, until getting under the threshold
    if rows_max_bytes is not None and rows_bytes >= rows_max_bytes:
        logger.debug(
            f"the size of the first {rows_min_number} rows ({rows_bytes}) is above the max number of bytes"
            f" ({rows_max_bytes}), they will be truncated"
        )
        return truncate_row_items(row_items, rows_max_bytes)

    # 3. else: add the remaining rows until the end, or until the bytes threshold
    for idx, row in enumerate(rows[rows_min_number:]):
        row_item = row.to_item()
        if rows_max_bytes is not None:
            rows_bytes += get_size_in_bytes(row_item)
            if rows_bytes >= rows_max_bytes:
                logger.debug(
                    f"the rows in the split have been truncated to {rows_min_number + idx} row(s) to keep the size"
                    f" ({rows_bytes}) under the limit ({rows_max_bytes})"
                )
                break
        row_items.append(row_item)
    return row_items


def get_rows_response(
    dataset_name: str,
    config_name: str,
    split_name: str,
    rows_max_bytes: Optional[int] = None,
    rows_min_number: Optional[int] = None,
) -> Tuple[Union[RowsResponse, None], Union[ErrorItem, None], int]:
    try:
        split = DbSplit.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).get()
    except DoesNotExist as e:
        raise Status400Error("Not found. Maybe the cache is missing, or maybe the split does not exist.", e) from e

    # ^ can also raise MultipleObjectsReturned, which should not occur -> we let the exception raise

    if split.status == Status.EMPTY:
        raise Status400Error("Not found. Cache is waiting to be refreshed.")
        # ^ should not occur with the current logic
    if split.status == Status.ERROR:
        split_error = DbSplitError.objects(
            dataset_name=dataset_name, config_name=config_name, split_name=split_name
        ).get()
        # ^ can raise DoesNotExist or MultipleObjectsReturned, which should not occur -> we let the exception raise
        return None, split_error.to_item(), split_error.status_code

    # TODO: if status is Status.STALLED, mention it in the response?
    columns = DbColumn.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).order_by(
        "+column_idx"
    )
    # TODO: on some datasets, such as "edbeeching/decision_transformer_gym_replay", it takes a long time, and we
    # truncate it anyway in to_row_items(). We might optimize here
    rows = DbRow.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).order_by(
        "+row_idx"
    )
    row_items = to_row_items(rows, rows_max_bytes, rows_min_number)
    rows_response: RowsResponse = {
        "columns": [column.to_item() for column in columns],
        "rows": row_items,
    }
    return rows_response, None, 200


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
    return [d.dataset_name for d in DbDataset.objects() if is_dataset_valid_or_stalled(d)]


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
