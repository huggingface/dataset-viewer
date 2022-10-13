# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import enum
import logging
import types
from datetime import datetime, timezone
from typing import Generic, List, Optional, Tuple, Type, TypedDict, TypeVar

from mongoengine import Document, DoesNotExist, connect
from mongoengine.errors import MultipleObjectsReturned
from mongoengine.fields import DateTimeField, EnumField, StringField
from mongoengine.queryset.queryset import QuerySet

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

logger = logging.getLogger(__name__)


class Status(enum.Enum):
    WAITING = "waiting"
    STARTED = "started"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


class JobDict(TypedDict):
    type: str
    dataset: str
    config: Optional[str]
    split: Optional[str]
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]


class CountByStatus(TypedDict):
    waiting: int
    started: int
    success: int
    error: int
    cancelled: int


# All the fields are optional
class DumpByStatus(TypedDict, total=False):
    waiting: List[JobDict]
    started: List[JobDict]
    success: List[JobDict]
    error: List[JobDict]
    cancelled: List[JobDict]


def connect_to_queue(database, host) -> None:
    connect(database, alias="queue", host=host)


# States:
# - waiting: started_at is None and finished_at is None: waiting jobs
# - started: started_at is not None and finished_at is None: started jobs
# - finished: started_at is not None and finished_at is not None: finished jobs
# - cancelled: cancelled_at is not None: cancelled jobs
# For a given set of arguments, any number of finished and cancelled jobs are allowed,
# but only 0 or 1 job for the set of the other states
class Job(Document):
    meta = {
        "collection": "jobs",
        "db_alias": "queue",
        "indexes": [
            "status",
            ("type", "status"),
            ("type", "dataset", "status"),
            ("type", "dataset", "config", "split", "status"),
        ],
    }
    type = StringField(required=True)
    dataset = StringField(required=True)
    config = StringField()
    split = StringField()
    created_at = DateTimeField(required=True)
    started_at = DateTimeField()
    finished_at = DateTimeField()
    status = EnumField(Status, default=Status.WAITING)

    def to_dict(self) -> JobDict:
        return {
            "type": self.type,
            "dataset": self.dataset,
            "config": self.config,
            "split": self.split,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def to_id(self) -> str:
        return f"Job[{self.type}][{self.dataset}][{self.config}][{self.split}]"

    objects = QuerySetManager["Job"]()


class EmptyQueue(Exception):
    pass


class JobNotFound(Exception):
    pass


def get_datetime() -> datetime:
    return datetime.now(timezone.utc)


def add_job(type: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None) -> Job:
    existing_jobs = Job.objects(type=type, dataset=dataset, config=config, split=split)
    new_job = Job(
        type=type, dataset=dataset, config=config, split=split, created_at=get_datetime(), status=Status.WAITING
    )
    pending_jobs = existing_jobs.filter(status__in=[Status.WAITING, Status.STARTED])
    try:
        # If one non-finished job exists, return it
        return pending_jobs.get()
    except DoesNotExist:
        # None exist, create one
        return new_job.save()
    except MultipleObjectsReturned:
        # should not happen, but it's not enforced in the database
        # (we could have one in WAITING status and another one in STARTED status)
        # if it happens, we "cancel" all of them, and re-run the same function
        pending_jobs.update(finished_at=get_datetime(), status=Status.CANCELLED)
        return add_job(type=type, dataset=dataset, config=config, split=split)


def get_jobs_with_status(status: Status, type: Optional[str] = None) -> QuerySet[Job]:
    if type is None:
        return Job.objects(status=status.value)
    return Job.objects(type=type, status=status.value)


def get_started(type: str) -> QuerySet[Job]:
    return get_jobs_with_status(status=Status.STARTED, type=type)


def get_num_started_for_dataset(dataset: str) -> int:
    return Job.objects(status=Status.STARTED, dataset=dataset).count()


def get_started_datasets(type: str) -> List[str]:
    return [job.dataset for job in Job.objects(type=type, status=Status.STARTED).only("dataset")]


def get_excluded_datasets(datasets: List[str], max_jobs_per_dataset: Optional[int] = None) -> List[str]:
    if max_jobs_per_dataset is None:
        return []
    return list({dataset for dataset in datasets if datasets.count(dataset) >= max_jobs_per_dataset})


def start_job(type: str, max_jobs_per_dataset: Optional[int] = None) -> Tuple[str, str, Optional[str], Optional[str]]:
    # try to get a job for a dataset that has still no started job
    started_datasets = get_started_datasets(type=type)
    next_waiting_job = (
        Job.objects(type=type, status=Status.WAITING, dataset__nin=started_datasets)
        .order_by("+created_at")
        .no_cache()
        .first()
    )
    # ^ no_cache should generate a query on every iteration, which should solve concurrency issues between workers
    if next_waiting_job is None:
        # the waiting jobs are all for datasets that already have started jobs.
        # let's take the next one, in the limit of max_jobs_per_dataset
        excluded_datasets = get_excluded_datasets(started_datasets, max_jobs_per_dataset)
        next_waiting_job = (
            Job.objects(type=type, status=Status.WAITING, dataset__nin=excluded_datasets)
            .order_by("+created_at")
            .no_cache()
            .first()
        )
    if next_waiting_job is None:
        raise EmptyQueue("no job available (within the limit of {max_jobs_per_dataset} started jobs per dataset)")
    next_waiting_job.update(started_at=get_datetime(), status=Status.STARTED)
    return str(next_waiting_job.pk), next_waiting_job.dataset, next_waiting_job.config, next_waiting_job.split
    # ^ job.pk is the id. job.id is not recognized by mypy


def finish_job(job_id: str, success: bool) -> None:
    try:
        job = Job.objects(pk=job_id).get()
    except DoesNotExist:
        logger.error(f"job {job_id} does not exist. Aborting.")
        return
    if job.status is not Status.STARTED:
        logger.warning(f"job {job.to_id()} has a not the STARTED status ({job.status.value}). Force finishing anyway.")
    if job.finished_at is not None:
        logger.warning(f"job {job.to_id()} has a non-empty finished_at field. Force finishing anyway.")
    if job.started_at is None:
        logger.warning(f"job {job.to_id()} has an empty started_at field. Force finishing anyway.")
    status = Status.SUCCESS if success else Status.ERROR
    job.update(finished_at=get_datetime(), status=status)


def cancel_started_jobs(type: str) -> None:
    for job in get_started(type=type):
        job.update(finished_at=get_datetime(), status=Status.CANCELLED)
        add_job(type=job.type, dataset=job.dataset, config=job.config, split=job.split)


def is_job_in_process(type: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None) -> bool:
    return (
        Job.objects(
            type=type, dataset=dataset, config=config, split=split, status__in=[Status.WAITING, Status.STARTED]
        ).count()
        > 0
    )


# special reports


def get_jobs_count_by_status(type: Optional[str] = None) -> CountByStatus:
    # ensure that all the statuses are present, even if equal to zero
    # note: we repeat the values instead of looping on Status because we don't know how to get the types right in mypy
    # result: CountByStatus = {s.value: jobs(status=s.value).count() for s in Status} # <- doesn't work in mypy
    # see https://stackoverflow.com/a/67292548/7351594
    return {
        "waiting": get_jobs_with_status(type=type, status=Status.WAITING).count(),
        "started": get_jobs_with_status(type=type, status=Status.STARTED).count(),
        "success": get_jobs_with_status(type=type, status=Status.SUCCESS).count(),
        "error": get_jobs_with_status(type=type, status=Status.ERROR).count(),
        "cancelled": get_jobs_with_status(type=type, status=Status.CANCELLED).count(),
    }


def get_dump_with_status(status: Status, type: Optional[str] = None) -> List[JobDict]:
    return [d.to_dict() for d in get_jobs_with_status(type=type, status=status)]


def get_dump_by_status(type: Optional[str] = None) -> DumpByStatus:
    return {
        "waiting": get_dump_with_status(type=type, status=Status.WAITING),
        "started": get_dump_with_status(type=type, status=Status.STARTED),
        "success": get_dump_with_status(type=type, status=Status.SUCCESS),
        "error": get_dump_with_status(type=type, status=Status.ERROR),
        "cancelled": get_dump_with_status(type=type, status=Status.CANCELLED),
    }


def get_dump_by_pending_status(type: Optional[str] = None) -> DumpByStatus:
    return {
        "waiting": get_dump_with_status(type=type, status=Status.WAITING),
        "started": get_dump_with_status(type=type, status=Status.STARTED),
    }


# only for the tests
def _clean_queue_database() -> None:
    Job.drop_collection()  # type: ignore


# explicit re-export
__all__ = ["DoesNotExist"]
