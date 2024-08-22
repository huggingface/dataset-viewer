# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import types
from typing import Generic, TypeVar

from mongoengine import Document
from mongoengine.fields import DateTimeField, StringField
from mongoengine.queryset.queryset import QuerySet

from libcommon.constants import (
    QUEUE_COLLECTION_DATASET_BLOCKAGES,
    QUEUE_MONGOENGINE_ALIAS,
)
from libcommon.utils import get_datetime

# START monkey patching ### hack ###
# see https://github.com/sbdchd/mongo-types#install
U = TypeVar("U", bound=Document)


def no_op(self, _):  # type: ignore
    return self


QuerySet.__class_getitem__ = types.MethodType(no_op, QuerySet)


class QuerySetManager(Generic[U]):
    def __get__(self, instance: object, cls: type[U]) -> QuerySet[U]:
        return QuerySet(cls, cls._get_collection())


# END monkey patching ### hack ###

# delete the dataset blockage (ie. potentially unblock it) after 6 hours
DATASET_BLOCKAGE_EXPIRE_AFTER_SECONDS = (
    6 * 60 * 60
)  # if we change this value, we have to ensure the TTL index is migrated accordingly

DATASET_STATUS_NORMAL = "normal"
DATASET_STATUS_BLOCKED = "blocked"


class DatasetBlockageDocument(Document):
    """A decision to block (rate-limit) a dataset. The same dataset can be blocked several times.
    It is released automatically when the blockage expires (DATASET_BLOCKAGE_EXPIRE_AFTER_SECONDS).

    Args:
        dataset (`str`): The dataset on which to apply the job.
        blocked_at (`datetime`): The date the dataset has been blocked.
          When read, it's an offset-naive datetime. Use pytz.UTC.localize() to make it timezone-aware.
    """

    meta = {
        "collection": QUEUE_COLLECTION_DATASET_BLOCKAGES,
        "db_alias": QUEUE_MONGOENGINE_ALIAS,
        "indexes": [
            ("dataset"),
            {
                "name": "DATASET_BLOCKAGE_EXPIRE_AFTER_SECONDS_BLUE",  # alternate BLUE/GREEN
                "fields": ["blocked_at"],
                "expireAfterSeconds": DATASET_BLOCKAGE_EXPIRE_AFTER_SECONDS,
            },
        ],
    }
    dataset = StringField(required=True)
    blocked_at = DateTimeField(required=True)

    objects = QuerySetManager["DatasetBlockageDocument"]()


def block_dataset(dataset: str) -> None:
    """Create a dataset blockage in the mongoDB database, at the current time.

    Args:
        dataset (`str`): The dataset to block.
    """
    DatasetBlockageDocument(dataset=dataset, blocked_at=get_datetime()).save()


def get_blocked_datasets() -> list[str]:
    """Return the list of blocked datasets."""
    return DatasetBlockageDocument.objects().distinct("dataset")


def is_blocked(dataset: str) -> bool:
    """Return True if the dataset is blocked."""
    return DatasetBlockageDocument.objects(dataset=dataset).count() > 0
