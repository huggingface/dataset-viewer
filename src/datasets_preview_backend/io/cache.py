import types
from typing import Any, Generic, List, Type, TypedDict, TypeVar, Union

from mongoengine import Document, DoesNotExist, connect
from mongoengine.fields import DictField, StringField
from mongoengine.queryset.queryset import QuerySet
from pymongo.errors import DocumentTooLarge

from datasets_preview_backend.config import MONGO_CACHE_DATABASE, MONGO_URL
from datasets_preview_backend.exceptions import (
    Status400Error,
    Status404Error,
    StatusErrorContent,
)
from datasets_preview_backend.models.dataset import Dataset

# START monkey patching ### hack ###
# see https://github.com/sbdchd/mongo-types#install
U = TypeVar("U", bound=Document)


def no_op(self, x):  # type: ignore
    return self


QuerySet.__class_getitem__ = types.MethodType(no_op, QuerySet)  # type: ignore


class QuerySetManager(Generic[U]):
    def __get__(self, instance: object, cls: Type[U]) -> QuerySet[U]:
        return QuerySet(cls, cls._get_collection())


# END monkey patching ### hack ###


def connect_to_cache() -> None:
    connect(MONGO_CACHE_DATABASE, alias="cache", host=MONGO_URL)


class DatasetCache(Document):
    meta = {"collection": "dataset_caches", "db_alias": "cache"}
    dataset_name = StringField(primary_key=True)
    status = StringField(required=True)  # TODO: enum
    content = DictField()  # TODO: more detail?

    objects = QuerySetManager["DatasetCache"]()


def get_dataset_cache(dataset_name: str) -> DatasetCache:
    try:
        return DatasetCache.objects(dataset_name=dataset_name).get()
    except DoesNotExist:
        # TODO: we might want to do a difference between:
        # - not found, but should exist because we checked with the list of datasets on HF (cache_miss)
        # - not found because this dataset does not exist
        # but both would return 404 anyway:
        # "The requested resource could not be found but may be available in the future.
        #  Subsequent requests by the client are permissible."
        return DatasetCache(
            dataset_name=dataset_name,
            status="cache_miss",
            content=Status404Error(
                "Not found. Maybe the cache is missing, or maybe the ressource does not exist."
            ).as_content(),
        )
    # it might also (but shouldn't) raise MultipleObjectsReturned: more than one result


def upsert_dataset_cache(dataset_name: str, status: str, content: Union[Dataset, StatusErrorContent]) -> None:
    try:
        DatasetCache(dataset_name=dataset_name, status=status, content=content).save()
    except DocumentTooLarge:
        # the content is over 16MB, see https://github.com/huggingface/datasets-preview-backend/issues/89
        DatasetCache(
            dataset_name=dataset_name,
            status="error",
            content=Status400Error(
                "The dataset document is larger than the maximum supported size (16MB)."
            ).as_content(),
        ).save()


def delete_dataset_cache(dataset_name: str) -> None:
    DatasetCache.objects(dataset_name=dataset_name).delete()


def clean_database() -> None:
    DatasetCache.drop_collection()  # type: ignore


# special reports


def get_dataset_names_with_status(status: str) -> List[str]:
    return [d.dataset_name for d in DatasetCache.objects(status=status).only("dataset_name")]


def get_datasets_count_with_status(status: str) -> int:
    return DatasetCache.objects(status=status).count()


class CacheReport(TypedDict):
    dataset: str
    status: str
    error: Union[Any, None]


def get_datasets_reports() -> List[CacheReport]:
    # first the valid entries: we don't want the content
    valid: List[CacheReport] = [
        {"dataset": d.dataset_name, "status": "valid", "error": None}
        for d in DatasetCache.objects(status="valid").only("dataset_name")
    ]

    # now the error entries
    error: List[CacheReport] = [
        {"dataset": d.dataset_name, "status": "error", "error": d.content}
        for d in DatasetCache.objects(status="error").only("dataset_name", "content")
    ]

    return valid + error
