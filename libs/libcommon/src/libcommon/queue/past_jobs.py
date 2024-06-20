# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import types
from datetime import datetime
from typing import Generic, TypeVar

from mongoengine import Document
from mongoengine.errors import ValidationError
from mongoengine.fields import DateTimeField, FloatField, StringField
from mongoengine.queryset.queryset import QuerySet

from libcommon.constants import (
    QUEUE_COLLECTION_PAST_JOBS,
    QUEUE_MONGOENGINE_ALIAS,
)

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

# delete the past jobs after 6 hours
PAST_JOB_EXPIRE_AFTER_SECONDS = 6 * 60 * 60


class PastJobDocument(Document):
    """The duration of a job that has been completed.

    Args:
        dataset (`str`): The dataset on which to apply the job.
        duration (`float`): The duration of the job, in seconds.
        finished_at (`datetime`): The date the job has finished.
    """

    meta = {
        "collection": QUEUE_COLLECTION_PAST_JOBS,
        "db_alias": QUEUE_MONGOENGINE_ALIAS,
        "indexes": [
            {
                "name": "PAST_JOB_EXPIRE_AFTER_SECONDS",
                "fields": ["finished_at"],
                "expireAfterSeconds": PAST_JOB_EXPIRE_AFTER_SECONDS,
            },
        ],
    }
    dataset = StringField(required=True)
    duration = FloatField(required=True, min_value=0.0)
    finished_at = DateTimeField(required=True)

    objects = QuerySetManager["PastJobDocument"]()


class NegativeDurationError(ValidationError):
    pass


def create_past_job(dataset: str, started_at: datetime, finished_at: datetime) -> None:
    """Create a past job in the mongoDB database

    Args:
        dataset (`str`): The dataset on which to apply the job.
        started_at (`datetime`): The date the job has started.
        finished_at (`datetime`): The date the job has finished.

    Raises:
        ValidationError: If the duration is negative.
    """
    duration = (finished_at - started_at).total_seconds()
    try:
        PastJobDocument(dataset=dataset, duration=duration, finished_at=finished_at).save()
    except ValidationError as e:
        raise NegativeDurationError("The duration of the job cannot be negative.") from e
