# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import types
from typing import Generic, TypeVar

from bson import ObjectId
from mongoengine import Document
from mongoengine.fields import DateTimeField, EnumField, IntField, ObjectIdField, StringField
from mongoengine.queryset.queryset import QuerySet

from libcommon.constants import (
    QUEUE_MONGOENGINE_ALIAS,
    TYPE_AND_STATUS_JOB_COUNTS_COLLECTION,
    WORKER_TYPE_JOB_COUNTS_COLLECTION,
)
from libcommon.dtos import Status, WorkerSize
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


class StartedJobError(Exception):
    pass


# END monkey patching ### hack ###


DEFAULT_INCREASE_AMOUNT = 1
DEFAULT_DECREASE_AMOUNT = -1


class JobTotalMetricDocument(Document):
    """Jobs total metric in mongoDB database, used to compute prometheus metrics.

    Args:
        job_type (`str`): job type
        status (`str`): job status see libcommon.queue.Status
        total (`int`): total of jobs
        created_at (`datetime`): when the metric has been created.
    """

    id = ObjectIdField(db_field="_id", primary_key=True, default=ObjectId)
    job_type = StringField(required=True, unique_with="status")
    status = StringField(required=True)
    total = IntField(required=True, default=0)
    created_at = DateTimeField(default=get_datetime)

    meta = {
        "collection": TYPE_AND_STATUS_JOB_COUNTS_COLLECTION,
        "db_alias": QUEUE_MONGOENGINE_ALIAS,
        "indexes": [("job_type", "status")],
    }
    objects = QuerySetManager["JobTotalMetricDocument"]()


class WorkerSizeJobsCountDocument(Document):
    """Metric that counts the number of (waiting) jobs per worker size.

    A worker size is defined by the job difficulties it handles. We hardcode
        - light: difficulty <= 40
        - medium: 40 < difficulty <= 70
        - heavy: 70 < difficulty

    Args:
        worker_size (`WorkerSize`): worker size
        jobs_count (`int`): jobs count
        created_at (`datetime`): when the metric has been created.
    """

    id = ObjectIdField(db_field="_id", primary_key=True, default=ObjectId)
    worker_size = EnumField(WorkerSize, required=True, unique=True)
    jobs_count = IntField(required=True, default=0)
    created_at = DateTimeField(default=get_datetime)

    @staticmethod
    def get_worker_size(difficulty: int) -> WorkerSize:
        if difficulty <= 40:
            return WorkerSize.light
        if difficulty <= 70:
            return WorkerSize.medium
        return WorkerSize.heavy

    meta = {
        "collection": WORKER_TYPE_JOB_COUNTS_COLLECTION,
        "db_alias": QUEUE_MONGOENGINE_ALIAS,
        "indexes": [("worker_size")],
    }
    objects = QuerySetManager["WorkerSizeJobsCountDocument"]()


def _update_metrics(job_type: str, status: str, increase_by: int, difficulty: int) -> None:
    JobTotalMetricDocument.objects(job_type=job_type, status=status).update(
        upsert=True,
        write_concern={"w": "majority", "fsync": True},
        read_concern={"level": "majority"},
        inc__total=increase_by,
    )
    if status == Status.WAITING:
        worker_size = WorkerSizeJobsCountDocument.get_worker_size(difficulty=difficulty)
        WorkerSizeJobsCountDocument.objects(worker_size=worker_size).update(
            upsert=True,
            write_concern={"w": "majority", "fsync": True},
            read_concern={"level": "majority"},
            inc__jobs_count=increase_by,
        )


def increase_metric(job_type: str, status: str, difficulty: int) -> None:
    _update_metrics(job_type=job_type, status=status, increase_by=DEFAULT_INCREASE_AMOUNT, difficulty=difficulty)


def decrease_metric(job_type: str, status: str, difficulty: int) -> None:
    _update_metrics(job_type=job_type, status=status, increase_by=DEFAULT_DECREASE_AMOUNT, difficulty=difficulty)


def update_metrics_for_type(job_type: str, previous_status: str, new_status: str, difficulty: int) -> None:
    if job_type is not None:
        decrease_metric(job_type=job_type, status=previous_status, difficulty=difficulty)
        increase_metric(job_type=job_type, status=new_status, difficulty=difficulty)
        # ^ this does not affect WorkerSizeJobsCountDocument, so we don't pass the job difficulty
