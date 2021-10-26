import types
from typing import Generic, Type, TypeVar

from mongoengine import Document, connect
from mongoengine.fields import DateTimeField, IntField, StringField
from mongoengine.queryset.queryset import QuerySet

from datasets_preview_backend.config import MONGO_CACHE_DATABASE, MONGO_URL

# from typing import Any, Generic, List, Type, TypedDict, TypeVar, Union


# from datasets_preview_backend.exceptions import Status404Error, StatusError
# from datasets_preview_backend.models.dataset import get_dataset

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


class Job(Document):
    dataset_name = StringField(primary_key=True)
    priority = IntField(required=True)
    start_time = DateTimeField(required=True)
    end_time = DateTimeField()

    objects = QuerySetManager["Job"]()


def connect_queue() -> None:
    connect(MONGO_CACHE_DATABASE, host=MONGO_URL)


def clean_database() -> None:
    Job.drop_collection()  # type: ignore
