import logging
import types
from typing import Any, Dict, Generic, List, Optional, Type, TypedDict, TypeVar, Union

from mongoengine import Document, DoesNotExist, connect
from mongoengine.fields import DictField, EnumField, IntField, ListField, StringField
from mongoengine.queryset.queryset import QuerySet
from pymongo.errors import DocumentTooLarge

from datasets_preview_backend.config import MONGO_CACHE_DATABASE, MONGO_URL
from datasets_preview_backend.exceptions import (
    Status400Error,
    Status404Error,
    Status500Error,
    StatusError,
)
from datasets_preview_backend.models.column import (
    ClassLabelColumn,
    ColumnDict,
    ColumnType,
)
from datasets_preview_backend.models.dataset import Dataset, get_dataset

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


# the purpose of this collection is to check if the dataset exists and which is it's status
class DbDataset(Document):
    dataset_name = StringField(required=True, unique=True)
    status = StringField(required=True)  # TODO: enum

    meta = {"collection": "datasets", "db_alias": "cache"}
    objects = QuerySetManager["DbDataset"]()


class SplitItem(TypedDict):
    dataset: str
    config: str
    split: str


class DbSplit(Document):
    dataset_name = StringField(required=True, unique_with=["config_name", "split_name"])
    config_name = StringField(required=True)
    split_name = StringField(required=True)

    def to_item(self) -> SplitItem:
        return {"dataset": self.dataset_name, "config": self.config_name, "split": self.split_name}

    meta = {"collection": "splits", "db_alias": "cache"}
    objects = QuerySetManager["DbSplit"]()


class RowItem(TypedDict):
    dataset: str
    config: str
    split: str
    row_idx: int
    row: Dict[str, Any]


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
        }

    meta = {"collection": "rows", "db_alias": "cache"}
    objects = QuerySetManager["DbRow"]()


class ColumnItem(TypedDict):
    dataset: str
    config: str
    split: str
    column_idx: int
    column: ColumnDict


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
    cause: str
    cause_exception: str
    cause_message: str


class DbError(Document):
    dataset_name = StringField(required=True, unique=True)
    status_code = IntField(required=True)  # TODO: an enum
    exception = StringField(required=True)
    message = StringField(required=True)
    cause_exception = StringField()
    cause_message = StringField()

    def to_item(self) -> ErrorItem:
        error: ErrorItem = {"status_code": self.status_code, "exception": self.exception, "message": self.message}
        if self.cause_exception and self.cause_message:
            error["cause"] = self.cause_exception  # to be deprecated
            error["cause_exception"] = self.cause_exception
            error["cause_message"] = self.cause_message
        return error

    meta = {"collection": "errors", "db_alias": "cache"}
    objects = QuerySetManager["DbError"]()


def upsert_error(dataset_name: str, error: StatusError) -> None:
    DbSplit.objects(dataset_name=dataset_name).delete()
    DbRow.objects(dataset_name=dataset_name).delete()
    DbColumn.objects(dataset_name=dataset_name).delete()
    DbDataset.objects(dataset_name=dataset_name).upsert_one(status="error")
    DbError.objects(dataset_name=dataset_name).upsert_one(
        status_code=error.status_code,
        exception=error.exception,
        message=error.message,
        cause_exception=error.cause_exception,
        cause_message=error.cause_message,
    )


def upsert_dataset(dataset: Dataset) -> None:
    dataset_name = dataset["dataset_name"]
    DbSplit.objects(dataset_name=dataset_name).delete()
    DbRow.objects(dataset_name=dataset_name).delete()
    DbColumn.objects(dataset_name=dataset_name).delete()
    DbError.objects(dataset_name=dataset_name).delete()
    try:
        DbDataset.objects(dataset_name=dataset_name).upsert_one(status="valid")
        for config in dataset["configs"]:
            config_name = config["config_name"]
            for split in config["splits"]:
                split_name = split["split_name"]
                rows = split["rows"]
                columns = split["columns"]
                DbSplit(dataset_name=dataset_name, config_name=config_name, split_name=split_name).save()
                for row_idx, row in enumerate(rows):
                    DbRow(
                        dataset_name=dataset_name,
                        config_name=config_name,
                        split_name=split_name,
                        row_idx=row_idx,
                        row=row,
                    ).save()
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
    except DocumentTooLarge:
        upsert_error(
            dataset_name, Status400Error("The dataset document is larger than the maximum supported size (16MB).")
        )
    except Exception as err:
        upsert_error(dataset_name, Status500Error(str(err)))


def delete_dataset_cache(dataset_name: str) -> None:
    DbDataset.objects(dataset_name=dataset_name).delete()
    DbSplit.objects(dataset_name=dataset_name).delete()
    DbRow.objects(dataset_name=dataset_name).delete()
    DbColumn.objects(dataset_name=dataset_name).delete()
    DbError.objects(dataset_name=dataset_name).delete()


def clean_database() -> None:
    DbDataset.drop_collection()  # type: ignore
    DbSplit.drop_collection()  # type: ignore
    DbRow.drop_collection()  # type: ignore
    DbColumn.drop_collection()  # type: ignore
    DbError.drop_collection()  # type: ignore


def refresh_dataset(dataset_name: str) -> None:
    try:
        dataset = get_dataset(dataset_name=dataset_name)
        upsert_dataset(dataset)
        logger.debug(f"dataset '{dataset_name}' is valid, cache updated")
    except StatusError as err:
        upsert_error(dataset_name, err)
        logger.debug(f"dataset '{dataset_name}' had error, cache updated")


# export


def check_dataset(dataset_name: str) -> None:
    if DbDataset.objects(dataset_name=dataset_name).count() == 0:
        raise Status404Error("Not found. Maybe the cache is missing, or maybe the dataset does not exist.")


def is_dataset_cached(dataset_name: str) -> bool:
    try:
        check_dataset(dataset_name)
    except Status404Error:
        return False
    return True


class NoError(Exception):
    pass


def get_error(dataset_name: str) -> ErrorItem:
    # can raise DoesNotExist
    try:
        return DbError.objects(dataset_name=dataset_name).get().to_item()
    except DoesNotExist:
        raise NoError()


def get_splits(
    dataset_name: str, config_name: Optional[str] = None, split_name: Optional[str] = None
) -> List[SplitItem]:
    if config_name is None:
        splits = DbSplit.objects(dataset_name=dataset_name)
    elif split_name is None:
        splits = DbSplit.objects(dataset_name=dataset_name, config_name=config_name)
    else:
        splits = DbSplit.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name)
    return [split.to_item() for split in splits]


def get_columns(
    dataset_name: str, config_name: Optional[str] = None, split_name: Optional[str] = None
) -> List[ColumnItem]:
    if config_name is None:
        columns = DbColumn.objects(dataset_name=dataset_name).order_by("+column_idx")
    elif split_name is None:
        columns = DbColumn.objects(dataset_name=dataset_name, config_name=config_name).order_by("+column_idx")
    else:
        columns = DbColumn.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).order_by(
            "+column_idx"
        )
    return [column.to_item() for column in columns]


def get_rows(dataset_name: str, config_name: Optional[str] = None, split_name: Optional[str] = None) -> List[RowItem]:
    if config_name is None:
        rows = DbRow.objects(dataset_name=dataset_name).order_by("+row_idx")
    elif split_name is None:
        rows = DbRow.objects(dataset_name=dataset_name, config_name=config_name).order_by("+row_idx")
    else:
        rows = DbRow.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name).order_by(
            "+row_idx"
        )
    return [row.to_item() for row in rows]


# special reports


def get_dataset_names_with_status(status: str) -> List[str]:
    return [d.dataset_name for d in DbDataset.objects(status=status).only("dataset_name")]


def get_datasets_count_with_status(status: str) -> int:
    return DbDataset.objects(status=status).count()


class CacheReport(TypedDict):
    dataset: str
    status: str
    error: Union[Any, None]


def get_datasets_reports() -> List[CacheReport]:
    # first the valid entries: we don't want the content
    valid: List[CacheReport] = [
        {"dataset": d.dataset_name, "status": "valid", "error": None}
        for d in DbDataset.objects(status="valid").only("dataset_name")
    ]

    # now the error entries
    error: List[CacheReport] = [
        {"dataset": error.dataset_name, "status": "error", "error": error.to_item()} for error in DbError.objects()
    ]

    return valid + error
