# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import enum
import logging
import types
from collections import Counter
from datetime import datetime, timezone
from itertools import groupby
from operator import itemgetter
from typing import Generic, List, Literal, Optional, Type, TypedDict, TypeVar

from mongoengine import Document, DoesNotExist, connect
from mongoengine.fields import BooleanField, DateTimeField, EnumField, StringField
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
    unicity_id: str
    namespace: str
    force: bool
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]


class StartedJobInfo(TypedDict):
    job_id: str
    dataset: str
    config: Optional[str]
    split: Optional[str]
    force: bool


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
# For a given set of arguments, only one job is allowed in the started state. No
# restriction for the other states
class Job(Document):
    """A job in the mongoDB database

    Args:
        type (`str`): The type of the job, identifies the queue
        dataset (`str`): The dataset on which to apply the job.
        config (`str`, optional): The config on which to apply the job.
        split (`str`, optional): The config on which to apply the job.
        unicity_id (`str`): A string that identifies the job uniquely. Only one job with the same unicity_id can be in
          the started state.
        namespace (`str`): The dataset namespace (user or organization) if any, else the dataset name (canonical name).
        force (`bool`, optional): If True, the job SHOULD not be skipped. Defaults to False.
        status (`Status`, optional): The status of the job. Defaults to Status.WAITING.
        created_at (`datetime`): The creation date of the job.
        started_at (`datetime`, optional): When the job has started.
        finished_at (`datetime`, optional): When the job has finished.
    """

    meta = {
        "collection": "jobsBlue",
        "db_alias": "queue",
        "indexes": [
            "status",
            ("type", "status"),
            ("type", "dataset", "status"),
            ("type", "dataset", "config", "split", "status"),
            ("status", "type", "created_at", "namespace"),
            "-created_at",
        ],
    }
    type = StringField(required=True)
    dataset = StringField(required=True)
    config = StringField()
    split = StringField()
    unicity_id = StringField(required=True)
    namespace = StringField(required=True)
    force = BooleanField(default=False)
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
            "unicity_id": self.unicity_id,
            "namespace": self.namespace,
            "force": self.force,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    objects = QuerySetManager["Job"]()


class Queue:
    """A queue manages jobs of a given type.

    Note that creating a Queue object does not create the queue in the database. It's a view that allows to manipulate
    the jobs. You can create multiple Queue objects, it has no effect on the database.

    It's a FIFO queue, with the following properties:
    - a job is identified by its input arguments: unicity_id (type, dataset, config and split)
    - a job can be in one of the following states: waiting, started, success, error, cancelled, skipped
    - a job can be in the queue only once (unicity_id) in the "started" state
    - a job can be in the queue multiple times in the other states (waiting, success, error, cancelled, skipped)
    - the queue is ordered by the creation date of the jobs
    - datasets and users that already have started jobs are de-prioritized (using namespace)
    - no more than `max_jobs_per_namespace` started jobs can exist for the same namespace

    Args:
        type (`str`, required): Type of the job. It identifies the queue.
        max_jobs_per_namespace (`int`): Maximum number of started jobs for the same namespace. We call a namespace the
          part of the dataset name that is before the `/` separator (user or organization). If `/` is not present,
          which is the case for the "canonical" datasets, the namespace is the dataset name.
          0 or a negative value are ignored. Defaults to None.
    """

    def __init__(self, type: str, max_jobs_per_namespace: Optional[int] = None):
        self.type = type
        self.max_jobs_per_namespace = (
            None if max_jobs_per_namespace is None or max_jobs_per_namespace < 1 else max_jobs_per_namespace
        )

    def add_job(self, dataset: str, config: Optional[str] = None, split: Optional[str] = None) -> Job:
        """Add a job to the queue in the waiting state.

        Returns: the job
        """
        return Job(
            type=self.type,
            dataset=dataset,
            config=config,
            split=split,
            unicity_id=f"Job[{self.type}][{dataset}][{config}][{split}]",
            namespace=dataset.split("/")[0],
            created_at=get_datetime(),
            status=Status.WAITING,
        ).save()

    def get_next_waiting_job(self) -> Job:
        """Get the next job in the queue.

        Get the waiting job with the oldest creation date:
        - first, among the datasets that still have no started job.
        - if none, among the datasets that have the least started jobs:
          - in the limit of `max_jobs_per_namespace` jobs per namespace
          - ensuring that the unicity_id field is unique among the started jobs.

        Raises:
            EmptyQueueError: if there is no waiting job in the queue that satisfies the restrictions above.

        Returns: the job
        """
        started_jobs = Job.objects(type=self.type, status=Status.STARTED)
        started_job_namespaces = [job.namespace for job in started_jobs.only("namespace")]

        next_waiting_job = (
            Job.objects(
                type=self.type,
                status=Status.WAITING,
                namespace__nin=set(started_job_namespaces),
            )
            .order_by("+created_at")
            .only("dataset", "config", "split")
            .no_cache()
            .first()
        )
        # ^ no_cache should generate a query on every iteration, which should solve concurrency issues between workers
        if next_waiting_job is not None:
            return next_waiting_job

        # all the waiting jobs, if any, are for namespaces that already have started jobs.
        #
        # Let's:
        # - exclude the waiting jobs for datasets that already have too many started jobs (max_jobs_per_namespace)
        # - exclude the waiting jobs which unicity_id is already in a started job
        # and, among the remaining waiting jobs, let's:
        # - select the oldest waiting job for the namespace with the least number of started jobs
        started_unicity_ids = {job.unicity_id for job in started_jobs.only("unicity_id")}
        descending_frequency_namespace_counts = [
            [namespace, count]
            for namespace, count in Counter(started_job_namespaces).most_common()
            if self.max_jobs_per_namespace is None or count < self.max_jobs_per_namespace
        ]
        descending_frequency_namespace_groups = [
            [item[0] for item in data] for (_, data) in groupby(descending_frequency_namespace_counts, itemgetter(1))
        ]
        # maybe we could get rid of this loop
        while descending_frequency_namespace_groups:
            least_common_namespaces_group = descending_frequency_namespace_groups.pop()
            next_waiting_job = (
                Job.objects(
                    type=self.type,
                    status=Status.WAITING,
                    namespace__in=least_common_namespaces_group,
                    unicity_id__nin=started_unicity_ids,
                )
                .order_by("+created_at")
                .only("dataset", "config", "split")
                .no_cache()
                .first()
            )
            if next_waiting_job is not None:
                return next_waiting_job
        raise EmptyQueueError(
            f"no job available (within the limit of {self.max_jobs_per_namespace} started jobs per namespace)"
        )

    def start_job(self) -> StartedJobInfo:
        """Start the next job in the queue.

        The job is moved from the waiting state to the started state.

        Raises:
            EmptyQueueError: if there is no job in the queue, within the limit of the maximum number of started jobs
            for a dataset

        Returns: the job id and the input arguments: dataset, config and split
        """
        next_waiting_job = self.get_next_waiting_job()
        # ^ can raise EmptyQueueError
        next_waiting_job.update(started_at=get_datetime(), status=Status.STARTED)
        return {
            "job_id": str(next_waiting_job.pk),  # job.pk is the id. job.id is not recognized by mypy
            "dataset": next_waiting_job.dataset,
            "config": next_waiting_job.config,
            "split": next_waiting_job.split,
            "force": next_waiting_job.force,
        }

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
                f"job {job.unicity_id} has a not the STARTED status ({job.status.value}). Force finishing anyway."
            )
        if job.finished_at is not None:
            logging.warning(f"job {job.unicity_id} has a non-empty finished_at field. Force finishing anyway.")
        if job.started_at is None:
            logging.warning(f"job {job.unicity_id} has an empty started_at field. Force finishing anyway.")
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
