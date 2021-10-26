import types
from typing import Generic, Type, TypeVar

from mongoengine import Document, DoesNotExist, connect
from mongoengine.fields import DictField, StringField
from mongoengine.queryset.queryset import QuerySet

from datasets_preview_backend.config import MONGO_CACHE_DATABASE, MONGO_URL
from datasets_preview_backend.exceptions import Status404Error, StatusError
from datasets_preview_backend.models.dataset import get_dataset

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


def connect_cache() -> None:
    connect(MONGO_CACHE_DATABASE, host=MONGO_URL)


class DatasetCache(Document):
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


def update_dataset_cache(dataset_name: str) -> None:
    try:
        dataset = get_dataset(dataset_name=dataset_name)
        dataset_cache = DatasetCache(dataset_name=dataset_name, status="valid", content=dataset)
    except StatusError as err:
        dataset_cache = DatasetCache(dataset_name=dataset_name, status="error", content=err.as_content())

    # https://docs.mongoengine.org/guide/validation.html#built-in-validation
    # dataset_cache.validate()     # raises ValidationError (Invalid email address: ['email'])
    dataset_cache.save()


def delete_dataset_cache(dataset_name: str) -> None:
    DatasetCache.objects(dataset_name=dataset_name).delete()


def clean_database() -> None:
    DatasetCache.drop_collection()  # type: ignore
