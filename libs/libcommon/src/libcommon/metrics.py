# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import types
from http import HTTPStatus
from typing import Generic, Type, TypeVar

from bson import ObjectId
from mongoengine import Document
from mongoengine.fields import (
    DateTimeField,
    EnumField,
    IntField,
    ObjectIdField,
    StringField,
)
from mongoengine.queryset.queryset import QuerySet

from libcommon.constants import METRICS_MONGOENGINE_ALIAS
from libcommon.utils import get_datetime

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


class JobTotalMetric(Document):
    """Jobs total metric in mongoDB database, used to compute prometheus metrics.

    Args:
        queue (`str`): queue name
        status (`str`): job status see libcommon.queue.Status
        total (`int`): total of jobs
        created_at (`datetime`): when the metric has been created.
    """

    id = ObjectIdField(db_field="_id", primary_key=True, default=ObjectId)
    queue = StringField(required=True)
    status = StringField(required=True)
    total = IntField(required=True, default=0)
    created_at = DateTimeField(default=get_datetime)

    meta = {
        "collection": "jobTotalMetric",
        "db_alias": METRICS_MONGOENGINE_ALIAS,
        "indexes": [("queue", "status")],
    }
    objects = QuerySetManager["JobTotalMetric"]()


class CacheTotalMetric(Document):
    """Cache total metric in the mongoDB database, used to compute prometheus metrics.

    Args:
        kind (`str`): kind name
        http_status (`str`): cache http_status
        error_code (`str`): error code name
        total (`int`): total of jobs
        created_at (`datetime`): when the metric has been created.
    """

    id = ObjectIdField(db_field="_id", primary_key=True, default=ObjectId)
    kind = StringField(required=True)
    http_status = EnumField(HTTPStatus, required=True)
    error_code = StringField()
    total = IntField(required=True, default=0)
    created_at = DateTimeField(default=get_datetime)

    meta = {
        "collection": "cacheTotalMetric",
        "db_alias": METRICS_MONGOENGINE_ALIAS,
        "indexes": [("kind", "http_status", "error_code")],
    }
    objects = QuerySetManager["CacheTotalMetric"]()
