# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import types
from datetime import datetime
from typing import Generic, TypeVar

from mongoengine import Document
from mongoengine.errors import ValidationError
from mongoengine.fields import DateTimeField, IntField, StringField
from mongoengine.queryset.queryset import QuerySet

from libcommon.constants import (
    QUEUE_COLLECTION_PAST_JOBS,
    QUEUE_MONGOENGINE_ALIAS,
)
from libcommon.queue.dataset_blockages import block_dataset, is_blocked

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

# we allow 10 hours of compute (parallel jobs) every hour, i.e. 10 dedicated machines
MAX_MACHINES = 10
# we look at the last 6 hours to decide to rate-limit a dataset
RATE_LIMIT_WINDOW_SECONDS = 6 * 60 * 60
# total jobs duration that triggers rate-limiting a dataset
DATASET_BLOCKAGE_THRESHOLD_SECONDS = MAX_MACHINES * RATE_LIMIT_WINDOW_SECONDS
# don't check for rate-limiting if the duration is super short
JOB_DURATION_CHECK_MIN_SECONDS = 5 * 60
# don't record short durations, because they will not have impact, but can clutter the collection
JOB_DURATION_MIN_SECONDS = 30


class PastJobDocument(Document):
    """The duration of a job that has been completed.

    Args:
        dataset (`str`): The dataset on which to apply the job.
        duration (`int`): The duration of the job, in seconds.
        finished_at (`datetime`): The date the job has finished.
    """

    meta = {
        "collection": QUEUE_COLLECTION_PAST_JOBS,
        "db_alias": QUEUE_MONGOENGINE_ALIAS,
        "indexes": [
            {
                "name": "PAST_JOB_EXPIRE_AFTER_SECONDS",
                "fields": ["finished_at"],
                "expireAfterSeconds": RATE_LIMIT_WINDOW_SECONDS,
            },
        ],
    }
    dataset = StringField(required=True)
    duration = IntField(required=True, min_value=0)
    finished_at = DateTimeField(required=True)

    objects = QuerySetManager["PastJobDocument"]()


class NegativeDurationError(ValidationError):
    pass


def create_past_job(dataset: str, started_at: datetime, finished_at: datetime) -> None:
    """Create a past job in the mongoDB database.

    After creating the entry, we check if it should be rate-limited (if it isn't yet), and if so, we block
    the dataset.

    Args:
        dataset (`str`): The dataset on which to apply the job.
        started_at (`datetime`): The date the job has started.
        finished_at (`datetime`): The date the job has finished.

    Raises:
        ValidationError: If the duration is negative.
    """
    duration = int((finished_at - started_at).total_seconds())
    if duration < JOB_DURATION_MIN_SECONDS:
        return
    try:
        PastJobDocument(dataset=dataset, duration=duration, finished_at=finished_at).save()
    except ValidationError as e:
        raise NegativeDurationError("The duration of the job cannot be negative.") from e

    if not is_blocked(dataset) and duration > JOB_DURATION_CHECK_MIN_SECONDS:
        if PastJobDocument.objects(dataset=dataset).sum("duration") > DATASET_BLOCKAGE_THRESHOLD_SECONDS:
            block_dataset(dataset)
