# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from mongoengine import Document
from mongoengine.fields import (
    DateTimeField,
    DictField,
    ObjectIdField,
    StringField,
)
from libcommon.constants import METRICS_MONGOENGINE_ALIAS, METRICS_TTL_SECONDS
from libcommon.utils import get_datetime
from bson import ObjectId
from typing import Generic, TypeVar, Type
from mongoengine.queryset.queryset import QuerySet
import types


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

class CustomMetric(Document):
    """A response computed for a job, cached in the mongoDB database

    Args:
        metric (`str`): metric name
        content (`dict`): The content of the metric.
        created_at (`datetime`): When the metric has been created.
    """

    id = ObjectIdField(db_field="_id", primary_key=True, default=ObjectId)

    metric = StringField(required=True, unique_with=["dataset", "config", "split"])
    content = DictField(required=True)
    created_at = DateTimeField(default=get_datetime)

    meta = {
        "collection": "customMetrics",
        "db_alias": METRICS_MONGOENGINE_ALIAS,
        "indexes": [
            ("metric", "created_at"),
            {"fields": ["created_at"], "expireAfterSeconds": METRICS_TTL_SECONDS},
        ],
    }
    objects = QuerySetManager["CustomMetric"]()
