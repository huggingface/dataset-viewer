# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import contextlib
import enum
import logging
import types
from collections import Counter
from datetime import datetime, timedelta, timezone
from itertools import groupby
from operator import itemgetter
from typing import Dict, Generic, List, Literal, Optional, Type, TypedDict, TypeVar

from mongoengine import Document, DoesNotExist
from mongoengine.fields import BooleanField, DateTimeField, EnumField, StringField
from mongoengine.queryset.queryset import QuerySet

from libcommon.constants import QUEUE_MONGOENGINE_ALIAS

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


class Priority(enum.Enum):
    NORMAL = "normal"
    LOW = "low"


class JobDict(TypedDict):
    type: str
    dataset: str
    config: Optional[str]
    split: Optional[str]
    unicity_id: str
    namespace: str
    force: bool
    priority: str
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]


class JobInfo(TypedDict):
    job_id: str
    type: str
    dataset: str
    config: Optional[str]
    split: Optional[str]
    force: bool
    priority: Priority


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
        split (`str`, optional): The split on which to apply the job.
        unicity_id (`str`): A string that identifies the job uniquely. Only one job with the same unicity_id can be in
          the started state.
        namespace (`str`): The dataset namespace (user or organization) if any, else the dataset name (canonical name).
        force (`bool`, optional): If True, the job SHOULD not be skipped. Defaults to False.
        priority (`Priority`, optional): The priority of the job. Defaults to Priority.NORMAL.
        status (`Status`, optional): The status of the job. Defaults to Status.WAITING.
        created_at (`datetime`): The creation date of the job.
        started_at (`datetime`, optional): When the job has started.
        finished_at (`datetime`, optional): When the job has finished.
    """

    meta = {
        "collection": "jobsBlue",
        "db_alias": QUEUE_MONGOENGINE_ALIAS,
        "indexes": [
            "dataset",
            "status",
            ("type", "status"),
            ("type", "dataset", "status"),
            ("type", "dataset", "config", "split", "status", "force", "priority"),
            ("priority", "status", "type", "created_at", "namespace", "unicity_id"),
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
    priority = EnumField(Priority, default=Priority.NORMAL)
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
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    objects = QuerySetManager["Job"]()


class Queue:
    """A queue manages jobs.

    Note that creating a Queue object does not create the queue in the database. It's a view that allows to manipulate
    the jobs. You can create multiple Queue objects, it has no effect on the database.

    It's a FIFO queue, with the following properties:
    - a job is identified by its input arguments: unicity_id (type, dataset, config and split)
    - a job can be in one of the following states: waiting, started, success, error, cancelled, skipped
    - a job can be in the queue only once (unicity_id) in the "started" or "waiting" state
    - a job can be in the queue multiple times in the other states (success, error, cancelled, skipped)
    - a job has a priority (two levels: NORMAL and LOW)
    - the queue is ordered by priority then by the creation date of the jobs
    - datasets and users that already have started jobs are de-prioritized (using namespace)
    - no more than `max_jobs_per_namespace` started jobs can exist for the same namespace

    Args:
        max_jobs_per_namespace (`int`): Maximum number of started jobs for the same namespace. We call a namespace the
          part of the dataset name that is before the `/` separator (user or organization). If `/` is not present,
          which is the case for the "canonical" datasets, the namespace is the dataset name.
          0 or a negative value are ignored. Defaults to None.
    """

    def __init__(self, max_jobs_per_namespace: Optional[int] = None):
        self.max_jobs_per_namespace = (
            None if max_jobs_per_namespace is None or max_jobs_per_namespace < 1 else max_jobs_per_namespace
        )

    def _add_job(
        self,
        job_type: str,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
        force: bool = False,
        priority: Priority = Priority.NORMAL,
    ) -> Job:
        """Add a job to the queue in the waiting state.

        This method should not be called directly. Use `upsert_job` instead.

        Args:
            job_type (`str`): The type of the job
            dataset (`str`): The dataset on which to apply the job.
            config (`str`, optional): The config on which to apply the job.
            split (`str`, optional): The config on which to apply the job.
            force (`bool`, optional): If True, the job SHOULD not be skipped. Defaults to False.
            priority (`Priority`, optional): The priority of the job. Defaults to Priority.NORMAL.

        Returns: the job
        """
        return Job(
            type=job_type,
            dataset=dataset,
            config=config,
            split=split,
            unicity_id=f"Job[{job_type}][{dataset}][{config}][{split}]",
            namespace=dataset.split("/")[0],
            force=force,
            priority=priority,
            created_at=get_datetime(),
            status=Status.WAITING,
        ).save()

    def upsert_job(
        self,
        job_type: str,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
        force: bool = False,
        priority: Priority = Priority.NORMAL,
    ) -> Job:
        """Add, or update, a job to the queue in the waiting state.

        If jobs already exist with the same parameters in the waiting state, they are cancelled and replaced by a new
        one.
        Note that the new job inherits the force=True property if one of the previous waiting jobs had it.
        In the same way, the new job inherits the highest priority.

        Args:
            job_type (`str`): The type of the job
            dataset (`str`): The dataset on which to apply the job.
            config (`str`, optional): The config on which to apply the job.
            split (`str`, optional): The config on which to apply the job.
            force (`bool`, optional): If True, the job SHOULD not be skipped. Defaults to False.
            priority (`Priority`, optional): The priority of the job. Defaults to Priority.NORMAL.

        Returns: the job
        """
        existing = Job.objects(type=job_type, dataset=dataset, config=config, split=split, status=Status.WAITING)
        if existing(force=True).count() > 0:
            force = True
        if existing(priority=Priority.NORMAL).count() > 0:
            priority = Priority.NORMAL
        existing.update(finished_at=get_datetime(), status=Status.CANCELLED)
        return self._add_job(
            job_type=job_type, dataset=dataset, config=config, split=split, force=force, priority=priority
        )

    def _get_next_waiting_job_for_priority(
        self, priority: Priority, only_job_types: Optional[list[str]] = None
    ) -> Job:
        """Get the next job in the queue for a given priority.

        For a given priority, get the waiting job with the oldest creation date:
        - among the datasets that still have no started job.
        - if none, among the datasets that have the least started jobs:
          - in the limit of `max_jobs_per_namespace` jobs per namespace
          - ensuring that the unicity_id field is unique among the started jobs.

        Args:
            priority (`Priority`): The priority of the job.
            only_job_types: if not None, only jobs of the given types are considered.

        Raises:
            EmptyQueueError: if there is no waiting job in the queue that satisfies the restrictions above.

        Returns: the job
        """
        logging.debug("Getting next waiting job for priority %s, among types %s", priority, only_job_types or "all")
        started_jobs = (
            Job.objects(type__in=only_job_types, status=Status.STARTED)
            if only_job_types
            else Job.objects(status=Status.STARTED)
        )
        logging.debug(f"Number of started jobs: {started_jobs.count()}")
        started_job_namespaces = [job.namespace for job in started_jobs.only("namespace")]
        logging.debug(f"Started job namespaces: {started_job_namespaces}")

        next_waiting_job = (
            (
                Job.objects(
                    type__in=only_job_types,
                    status=Status.WAITING,
                    namespace__nin=set(started_job_namespaces),
                    priority=priority,
                )
                if only_job_types
                else Job.objects(
                    status=Status.WAITING,
                    namespace__nin=set(started_job_namespaces),
                    priority=priority,
                )
            )
            .order_by("+created_at")
            .only("type", "dataset", "config", "split", "force")
            .no_cache()
            .first()
        )
        # ^ no_cache should generate a query on every iteration, which should solve concurrency issues between workers
        if next_waiting_job is not None:
            return next_waiting_job
        logging.debug("No waiting job for namespace without started job")

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
        logging.debug(
            f"Descending frequency namespace counts, with less than {self.max_jobs_per_namespace} started jobs:"
            f" {descending_frequency_namespace_counts}"
        )
        descending_frequency_namespace_groups = [
            [item[0] for item in data] for (_, data) in groupby(descending_frequency_namespace_counts, itemgetter(1))
        ]
        # maybe we could get rid of this loop
        while descending_frequency_namespace_groups:
            least_common_namespaces_group = descending_frequency_namespace_groups.pop()
            logging.debug(f"Least common namespaces group: {least_common_namespaces_group}")
            next_waiting_job = (
                (
                    Job.objects(
                        type__in=only_job_types,
                        status=Status.WAITING,
                        namespace__in=least_common_namespaces_group,
                        unicity_id__nin=started_unicity_ids,
                        priority=priority,
                    )
                    if only_job_types
                    else Job.objects(
                        status=Status.WAITING,
                        namespace__in=least_common_namespaces_group,
                        unicity_id__nin=started_unicity_ids,
                        priority=priority,
                    )
                )
                .order_by("+created_at")
                .only("type", "dataset", "config", "split", "force")
                .no_cache()
                .first()
            )
            if next_waiting_job is not None:
                return next_waiting_job
        raise EmptyQueueError(
            f"no job available with the priority (within the limit of {self.max_jobs_per_namespace} started jobs per"
            " namespace)"
        )

    def get_next_waiting_job(self, only_job_types: Optional[list[str]] = None) -> Job:
        """Get the next job in the queue.

        Get the waiting job with the oldest creation date with the following criteria:
        - among the highest priority jobs,
        - among the datasets that still have no started job.
        - if none, among the datasets that have the least started jobs:
          - in the limit of `max_jobs_per_namespace` jobs per namespace
          - ensuring that the unicity_id field is unique among the started jobs.

        Args:
            only_job_types: if not None, only jobs of the given types are considered.

        Raises:
            EmptyQueueError: if there is no waiting job in the queue that satisfies the restrictions above.

        Returns: the job
        """
        for priority in [Priority.NORMAL, Priority.LOW]:
            with contextlib.suppress(EmptyQueueError):
                return self._get_next_waiting_job_for_priority(priority=priority, only_job_types=only_job_types)
        raise EmptyQueueError(
            f"no job available (within the limit of {self.max_jobs_per_namespace} started jobs per namespace)"
        )

    def start_job(self, only_job_types: Optional[list[str]] = None) -> JobInfo:
        """Start the next job in the queue.

        The job is moved from the waiting state to the started state.

        Args:
            only_job_types: if not None, only jobs of the given types are considered.

        Raises:
            EmptyQueueError: if there is no job in the queue, within the limit of the maximum number of started jobs
            for a dataset

        Returns: the job id, the type, the input arguments: dataset, config and split and the force flag
        """
        logging.debug("looking for a job to start, among the following types: %s", only_job_types or "all")
        next_waiting_job = self.get_next_waiting_job(only_job_types=only_job_types)
        logging.debug(f"job found: {next_waiting_job}")
        # ^ can raise EmptyQueueError
        next_waiting_job.update(started_at=get_datetime(), status=Status.STARTED)
        if only_job_types and next_waiting_job.type not in only_job_types:
            raise RuntimeError(
                f"The job type {next_waiting_job.type} is not in the list of allowed job types {only_job_types}"
            )
        return {
            "job_id": str(next_waiting_job.pk),  # job.pk is the id. job.id is not recognized by mypy
            "type": next_waiting_job.type,
            "dataset": next_waiting_job.dataset,
            "config": next_waiting_job.config,
            "split": next_waiting_job.split,
            "force": next_waiting_job.force,
            "priority": next_waiting_job.priority,
        }

    def get_job_type(self, job_id: str) -> str:
        """Get the job type for a given job id.

        Args:
            job_id (`str`, required): id of the job

        Returns: the job type

        Raises:
            DoesNotExist: if the job does not exist
        """
        job = Job.objects(pk=job_id).get()
        return job.type

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

    def is_job_in_process(
        self, job_type: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None
    ) -> bool:
        """Check if a job is in process (waiting or started).

        Args:
            job_type (`str`, required): job type
            dataset (`str`, required): dataset name
            config (`str`, optional): config name. Defaults to None.
            split (`str`, optional): split name. Defaults to None.

        Returns:
            `bool`: whether the job is in process (waiting or started)
        """
        return (
            Job.objects(
                type=job_type,
                dataset=dataset,
                config=config,
                split=split,
                status__in=[Status.WAITING, Status.STARTED],
            ).count()
            > 0
        )

    def cancel_started_jobs(self, job_type: str) -> None:
        """Cancel all started jobs for a given type."""
        for job in Job.objects(type=job_type, status=Status.STARTED.value):
            job.update(finished_at=get_datetime(), status=Status.CANCELLED)
            self.upsert_job(job_type=job.type, dataset=job.dataset, config=job.config, split=job.split)

    # special reports
    def count_jobs(self, status: Status, job_type: str) -> int:
        """Count the number of jobs with a given status and the given type.

        Args:
            status (`Status`, required): status of the jobs
            job_type (`str`, required): job type

        Returns: the number of jobs with the given status and the given type.
        """
        return Job.objects(type=job_type, status=status.value).count()

    def get_jobs_count_by_status(self, job_type: str) -> CountByStatus:
        """Count the number of jobs by status for a given job type.

        Returns: a dictionary with the number of jobs for each status
        """
        # ensure that all the statuses are present, even if equal to zero
        # note: we repeat the values instead of looping on Status because we don't know how to get the types right
        # in mypy
        # result: CountByStatus = {s.value: jobs(status=s.value).count() for s in Status} # <- doesn't work in mypy
        # see https://stackoverflow.com/a/67292548/7351594
        return {
            "waiting": self.count_jobs(status=Status.WAITING, job_type=job_type),
            "started": self.count_jobs(status=Status.STARTED, job_type=job_type),
            "success": self.count_jobs(status=Status.SUCCESS, job_type=job_type),
            "error": self.count_jobs(status=Status.ERROR, job_type=job_type),
            "cancelled": self.count_jobs(status=Status.CANCELLED, job_type=job_type),
            "skipped": self.count_jobs(status=Status.SKIPPED, job_type=job_type),
        }

    def get_dump_with_status(self, status: Status, job_type: str) -> List[JobDict]:
        """Get the dump of the jobs with a given status and a given type.

        Args:
            status (`Status`, required): status of the jobs
            job_type (`str`, required): job type

        Returns: a list of jobs with the given status and the given type
        """
        return [d.to_dict() for d in Job.objects(type=job_type, status=status.value)]

    def get_dump_by_pending_status(self, job_type: str) -> DumpByPendingStatus:
        """Get the dump of the jobs by pending status for a given job type.

        Returns: a dictionary with the dump of the jobs for each pending status
        """
        return {
            "waiting": self.get_dump_with_status(job_type=job_type, status=Status.WAITING),
            "started": self.get_dump_with_status(job_type=job_type, status=Status.STARTED),
        }

    def get_total_duration_per_dataset(self, job_type: str) -> Dict[str, int]:
        """Get the total duration for the last 30 days of the finished jobs of a given type for every dataset

        Returns: a dictionary where the keys are the dataset names and the values are the total duration of its
        finished jobs during the last 30 days, in seconds (integer)
        """
        DURATION_IN_DAYS = 30
        return {
            d["_id"]: d["total_duration"]
            for d in Job.objects(
                type=job_type,
                status__in=[Status.SUCCESS, Status.ERROR],
                finished_at__gt=datetime.now() - timedelta(days=DURATION_IN_DAYS),
            ).aggregate(
                {
                    "$group": {
                        "_id": "$dataset",
                        "total_duration": {
                            "$sum": {
                                "$dateDiff": {"startDate": "$started_at", "endDate": "$finished_at", "unit": "second"}
                            }
                        },
                    }
                }
            )
        }


# only for the tests
def _clean_queue_database() -> None:
    """Delete all the jobs in the database"""
    Job.drop_collection()  # type: ignore


# explicit re-export
__all__ = ["DoesNotExist"]
