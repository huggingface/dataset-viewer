# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import enum
import logging
import types
from datetime import datetime, timezone
from typing import Generic, List, Literal, Optional, Tuple, Type, TypedDict, TypeVar

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


class Status(enum.Enum):
    WAITING = "waiting"
    STARTED = "started"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


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
    skipped: int


class DumpByPendingStatus(TypedDict):
    waiting: List[JobDict]
    started: List[JobDict]


class EmptyQueueError(Exception):
    pass


def get_datetime() -> datetime:
    return datetime.now(timezone.utc)


def connect_to_database(database: str, host: str) -> None:
    connect(db=database, alias="queue", host=host)


# States:
# - waiting: started_at is None and finished_at is None: waiting jobs
# - started: started_at is not None and finished_at is None: started jobs
# - finished: started_at is not None and finished_at is not None: finished jobs
# For a given set of arguments, any number of finished and cancelled jobs are allowed,
# but only 0 or 1 job for the set of the other states
class Job(Document):
    """A job in the mongoDB database

    Args:
        type (`str`): The type of the job, identifies the queue
        dataset (`str`): The dataset on which to apply the job.
        config (`str`, optional): The config on which to apply the job.
        split (`str`, optional): The config on which to apply the job.
        status (`Status`, optional): The status of the job. Defaults to Status.WAITING.
        created_at (`datetime`): The creation date of the job.
        started_at (`datetime`, optional): When the job has started.
        finished_at (`datetime`, optional): When the job has finished.
    """

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
    status = EnumField(Status, default=Status.WAITING)
    created_at = DateTimeField(required=True)
    started_at = DateTimeField()
    finished_at = DateTimeField()

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


class Queue:
    """A queue manages jobs of a given type.

    Note that creating a Queue object does not create the queue in the database. It's a view that allows to manipulate
    the jobs. You can create multiple Queue objects, it has no effect on the database.

    It's a FIFO queue, with the following properties:
    - a job is identified by its input arguments: dataset, and optionally config and split
    - a job can be in one of the following states: waiting, started, success, error, cancelled
    - a job can be in the queue only once in a pending state (waiting or started)
    - a job can be in the queue multiple times in a finished state (success, error, cancelled)
    - the queue is ordered by the creation date of the jobs
    - datasets that already have started job are de-prioritized
    - datasets cannot have more than `max_jobs_per_dataset` started jobs

    Args:
        type (`str`, required): Type of the job. It identifies the queue.
        max_jobs_per_dataset (`int`): Maximum number of started jobs for the same dataset. 0 or a negative value
          are ignored. Defaults to None.
    """

    def __init__(self, type: str, max_jobs_per_dataset: Optional[int] = None):
        self.type = type
        self.max_jobs_per_dataset = (
            None if max_jobs_per_dataset is None or max_jobs_per_dataset < 1 else max_jobs_per_dataset
        )

    def add_job(self, dataset: str, config: Optional[str] = None, split: Optional[str] = None) -> Job:
        """Add a job to the queue in the waiting state.

        If a job with the same arguments already exists in the queue in a pending state (waiting, started), no new job
        is created and the existing job is returned.

        Returns: the job
        """
        existing_jobs = Job.objects(type=self.type, dataset=dataset, config=config, split=split)
        new_job = Job(
            type=self.type,
            dataset=dataset,
            config=config,
            split=split,
            created_at=get_datetime(),
            status=Status.WAITING,
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
            return self.add_job(dataset=dataset, config=config, split=split)

    def start_job(self) -> Tuple[str, str, Optional[str], Optional[str]]:
        """Start the next job in the queue.

        Get the next job in the queue, among the datasets that still have no started job.
        If no job is available, get the next job in the queue, among the datasets that already have a started job,
        but not more than `max_jobs_per_dataset` jobs per dataset.

        The job is moved from the waiting state to the started state.

        Raises:
            EmptyQueueError: if there is no job in the queue, within the limit of the maximum number of started jobs
            for a dataset

        Returns: the job id and the input arguments: dataset, config and split
        """
        # try to get a job for a dataset that still has no started job
        started_datasets = [job.dataset for job in Job.objects(type=self.type, status=Status.STARTED).only("dataset")]
        next_waiting_job = (
            Job.objects(type=self.type, status=Status.WAITING, dataset__nin=started_datasets)
            .order_by("+created_at")
            .no_cache()
            .first()
        )
        # ^ no_cache should generate a query on every iteration, which should solve concurrency issues between workers
        if next_waiting_job is None:
            # the waiting jobs are all for datasets that already have started jobs.
            # let's take the next one, in the limit of max_jobs_per_dataset
            excluded_datasets = (
                []
                if self.max_jobs_per_dataset is None
                else list(
                    {
                        dataset
                        for dataset in started_datasets
                        if started_datasets.count(dataset) >= self.max_jobs_per_dataset
                    }
                )
            )
            next_waiting_job = (
                Job.objects(type=self.type, status=Status.WAITING, dataset__nin=excluded_datasets)
                .order_by("+created_at")
                .no_cache()
                .first()
            )
        if next_waiting_job is None:
            raise EmptyQueueError(
                "no job available (within the limit of {max_jobs_per_dataset} started jobs per dataset)"
            )
        next_waiting_job.update(started_at=get_datetime(), status=Status.STARTED)
        return str(next_waiting_job.pk), next_waiting_job.dataset, next_waiting_job.config, next_waiting_job.split
        # ^ job.pk is the id. job.id is not recognized by mypy

    def finish_job(self, job_id: str, finished_status: Literal[Status.SUCCESS, Status.ERROR, Status.SKIPPED]) -> None:
        """Finish a job in the queue.

        The job is moved from the started state to the success or error state.

        Args:
            job_id (`str`, required): id of the job
            success (`bool`, required): whether the job succeeded or not

        Returns: nothing
        """
        try:
            job = Job.objects(pk=job_id).get()
        except DoesNotExist:
            logging.error(f"job {job_id} does not exist. Aborting.")
            return
        if job.status is not Status.STARTED:
            logging.warning(
                f"job {job.to_id()} has a not the STARTED status ({job.status.value}). Force finishing anyway."
            )
        if job.finished_at is not None:
            logging.warning(f"job {job.to_id()} has a non-empty finished_at field. Force finishing anyway.")
        if job.started_at is None:
            logging.warning(f"job {job.to_id()} has an empty started_at field. Force finishing anyway.")
        job.update(finished_at=get_datetime(), status=finished_status)

    def is_job_in_process(self, dataset: str, config: Optional[str] = None, split: Optional[str] = None) -> bool:
        """Check if a job is in process (waiting or started).

        Args:
            dataset (`str`, required): dataset name
            config (`str`, optional): config name. Defaults to None.
            split (`str`, optional): split name. Defaults to None.

        Returns:
            `bool`: whether the job is in process (waiting or started)
        """
        return (
            Job.objects(
                type=self.type,
                dataset=dataset,
                config=config,
                split=split,
                status__in=[Status.WAITING, Status.STARTED],
            ).count()
            > 0
        )

    def cancel_started_jobs(self) -> None:
        """Cancel all started jobs."""
        for job in Job.objects(type=self.type, status=Status.STARTED.value):
            job.update(finished_at=get_datetime(), status=Status.CANCELLED)
            self.add_job(dataset=job.dataset, config=job.config, split=job.split)

    # special reports
    def count_jobs(self, status: Status) -> int:
        """Count the number of jobs with a given status.

        Args:
            status (`Status`, required): status of the jobs

        Returns: the number of jobs with the given status
        """
        return Job.objects(type=self.type, status=status.value).count()

    def get_jobs_count_by_status(self) -> CountByStatus:
        """Count the number of jobs by status.

        Returns: a dictionary with the number of jobs for each status
        """
        # ensure that all the statuses are present, even if equal to zero
        # note: we repeat the values instead of looping on Status because we don't know how to get the types right
        # in mypy
        # result: CountByStatus = {s.value: jobs(status=s.value).count() for s in Status} # <- doesn't work in mypy
        # see https://stackoverflow.com/a/67292548/7351594
        return {
            "waiting": self.count_jobs(status=Status.WAITING),
            "started": self.count_jobs(status=Status.STARTED),
            "success": self.count_jobs(status=Status.SUCCESS),
            "error": self.count_jobs(status=Status.ERROR),
            "cancelled": self.count_jobs(status=Status.CANCELLED),
            "skipped": self.count_jobs(status=Status.SKIPPED),
        }

    def get_dump_with_status(self, status: Status) -> List[JobDict]:
        """Get the dump of the jobs with a given status.

        Args:
            status (`Status`, required): status of the jobs

        Returns: a list of jobs with the given status
        """
        return [d.to_dict() for d in Job.objects(type=self.type, status=status.value)]

    def get_dump_by_pending_status(self) -> DumpByPendingStatus:
        """Get the dump of the jobs by pending status.

        Returns: a dictionary with the dump of the jobs for each pending status
        """
        return {
            "waiting": self.get_dump_with_status(status=Status.WAITING),
            "started": self.get_dump_with_status(status=Status.STARTED),
        }


# only for the tests
def _clean_queue_database() -> None:
    """Delete all the jobs in the database"""
    Job.drop_collection()  # type: ignore


# explicit re-export
__all__ = ["DoesNotExist"]
