# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import contextlib
import logging
import types
from collections import Counter
from datetime import datetime, timedelta
from itertools import groupby
from operator import itemgetter
from typing import Generic, List, Optional, Type, TypedDict, TypeVar

import pytz
from mongoengine import Document, DoesNotExist
from mongoengine.fields import DateTimeField, EnumField, StringField
from mongoengine.queryset.queryset import QuerySet

from libcommon.constants import (
    QUEUE_COLLECTION_JOBS,
    QUEUE_MONGOENGINE_ALIAS,
    QUEUE_TTL_SECONDS,
)
from libcommon.utils import JobInfo, Priority, Status, get_datetime, inputs_to_string

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


class JobDict(TypedDict):
    type: str
    dataset: str
    revision: str
    config: Optional[str]
    split: Optional[str]
    unicity_id: str
    namespace: str
    priority: str
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    last_heartbeat: Optional[datetime]


class CountByStatus(TypedDict):
    waiting: int
    started: int
    success: int
    error: int
    cancelled: int


class DumpByPendingStatus(TypedDict):
    waiting: List[JobDict]
    started: List[JobDict]


class EmptyQueueError(Exception):
    pass


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
        revision (`str`): The git revision of the dataset.
        config (`str`, optional): The config on which to apply the job.
        split (`str`, optional): The split on which to apply the job.
        unicity_id (`str`): A string that identifies the job uniquely. Only one job with the same unicity_id can be in
          the started state. The revision is not part of the unicity_id.
        namespace (`str`): The dataset namespace (user or organization) if any, else the dataset name (canonical name).
        priority (`Priority`, optional): The priority of the job. Defaults to Priority.NORMAL.
        status (`Status`, optional): The status of the job. Defaults to Status.WAITING.
        created_at (`datetime`): The creation date of the job.
        started_at (`datetime`, optional): When the job has started.
        finished_at (`datetime`, optional): When the job has finished.
        last_heartbeat (`datetime`, optional): Last time the running job got a heartbeat from the worker.
    """

    meta = {
        "collection": QUEUE_COLLECTION_JOBS,
        "db_alias": QUEUE_MONGOENGINE_ALIAS,
        "indexes": [
            "dataset",
            "status",
            ("type", "status"),
            ("type", "dataset", "status"),
            ("type", "dataset", "revision", "config", "split", "status", "priority"),
            ("priority", "status", "created_at", "namespace", "unicity_id"),
            ("priority", "status", "created_at", "type", "namespace"),
            ("priority", "status", "type", "created_at", "namespace", "unicity_id"),
            ("priority", "status", "created_at", "namespace", "type", "unicity_id"),
            ("status", "type"),
            ("status", "namespace", "priority", "type", "created_at"),
            ("status", "namespace", "unicity_id", "priority", "type", "created_at"),
            "-created_at",
            {"fields": ["finished_at"], "expireAfterSeconds": QUEUE_TTL_SECONDS},
        ],
    }
    type = StringField(required=True)
    dataset = StringField(required=True)
    revision = StringField(required=True)
    config = StringField()
    split = StringField()
    unicity_id = StringField(required=True)
    namespace = StringField(required=True)
    priority = EnumField(Priority, default=Priority.NORMAL)
    status = EnumField(Status, default=Status.WAITING)
    created_at = DateTimeField(required=True)
    started_at = DateTimeField()
    finished_at = DateTimeField()
    last_heartbeat = DateTimeField()

    def to_dict(self) -> JobDict:
        return {
            "type": self.type,
            "dataset": self.dataset,
            "revision": self.revision,
            "config": self.config,
            "split": self.split,
            "unicity_id": self.unicity_id,
            "namespace": self.namespace,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_heartbeat": self.last_heartbeat,
        }

    objects = QuerySetManager["Job"]()

    def info(self) -> JobInfo:
        return JobInfo(
            {
                "job_id": str(self.pk),  # job.pk is the id. job.id is not recognized by mypy
                "type": self.type,
                "params": {
                    "dataset": self.dataset,
                    "revision": self.revision,
                    "config": self.config,
                    "split": self.split,
                },
                "priority": self.priority,
            }
        )


class Queue:
    """A queue manages jobs.

    Note that creating a Queue object does not create the queue in the database. It's a view that allows to manipulate
    the jobs. You can create multiple Queue objects, it has no effect on the database.

    It's a FIFO queue, with the following properties:
    - a job is identified by its input arguments: unicity_id (type, dataset, config and split, NOT revision)
    - a job can be in one of the following states: waiting, started, success, error, cancelled
    - a job can be in the queue only once (unicity_id) in the "started" or "waiting" state
    - a job can be in the queue multiple times in the other states (success, error, cancelled)
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

    def __init__(
        self,
        max_jobs_per_namespace: Optional[int] = None,
    ):
        self.max_jobs_per_namespace = (
            None if max_jobs_per_namespace is None or max_jobs_per_namespace < 1 else max_jobs_per_namespace
        )

    def _add_job(
        self,
        job_type: str,
        dataset: str,
        revision: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
        priority: Priority = Priority.NORMAL,
    ) -> Job:
        """Add a job to the queue in the waiting state.

        This method should not be called directly. Use `upsert_job` instead.

        Args:
            job_type (`str`): The type of the job
            dataset (`str`): The dataset on which to apply the job.
            revision (`str`): The git revision of the dataset.
            config (`str`, optional): The config on which to apply the job.
            split (`str`, optional): The config on which to apply the job.
            priority (`Priority`, optional): The priority of the job. Defaults to Priority.NORMAL.

        Returns: the job
        """
        return Job(
            type=job_type,
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            unicity_id=inputs_to_string(dataset=dataset, config=config, split=split, prefix=job_type),
            namespace=dataset.split("/")[0],
            priority=priority,
            created_at=get_datetime(),
            status=Status.WAITING,
        ).save()

    def upsert_job(
        self,
        job_type: str,
        dataset: str,
        revision: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
        priority: Priority = Priority.NORMAL,
    ) -> Job:
        """Add, or update, a job to the queue in the waiting state.

        If jobs already exist with the same parameters in the waiting state, they are cancelled and replaced by a new
        one.
        Note that the new job inherits the highest priority of the previous waiting jobs.

        Args:
            job_type (`str`): The type of the job
            dataset (`str`): The dataset on which to apply the job.
            revision (`str`): The git revision of the dataset.
            config (`str`, optional): The config on which to apply the job.
            split (`str`, optional): The config on which to apply the job.
            priority (`Priority`, optional): The priority of the job. Defaults to Priority.NORMAL.

        Returns: the job
        """
        canceled_jobs = self.cancel_jobs(
            job_type=job_type,
            dataset=dataset,
            config=config,
            split=split,
            statuses_to_cancel=[Status.WAITING],
        )
        if any(job["priority"] == Priority.NORMAL for job in canceled_jobs):
            priority = Priority.NORMAL
        return self._add_job(
            job_type=job_type, dataset=dataset, revision=revision, config=config, split=split, priority=priority
        )

    def cancel_jobs(
        self,
        job_type: str,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
        statuses_to_cancel: Optional[List[Status]] = None,
    ) -> List[JobDict]:
        """Cancel jobs from the queue.

        Note that the jobs for all the revisions are canceled.

        Returns the list of canceled jobs (as JobDict, before they are canceled, to be able to know their previous
        status)

        Args:
            job_type (`str`): The type of the job
            dataset (`str`): The dataset on which to apply the job.
            config (`str`, optional): The config on which to apply the job.
            split (`str`, optional): The config on which to apply the job.
            statuses_to_cancel (`list[Status]`, optional): The list of statuses to cancel. Defaults to
                [Status.WAITING, Status.STARTED].

        Returns:
            `list[JobDict]`: The list of canceled jobs
        """
        if statuses_to_cancel is None:
            statuses_to_cancel = [Status.WAITING, Status.STARTED]
        existing = Job.objects(
            type=job_type,
            dataset=dataset,
            config=config,
            split=split,
            status__in=statuses_to_cancel,
        )
        job_dicts = [job.to_dict() for job in existing]
        existing.update(finished_at=get_datetime(), status=Status.CANCELLED)
        return job_dicts

    def _get_next_waiting_job_for_priority(
        self,
        priority: Priority,
        job_types_blocked: Optional[list[str]] = None,
        job_types_only: Optional[list[str]] = None,
    ) -> Job:
        """Get the next job in the queue for a given priority.

        For a given priority, get the waiting job with the oldest creation date:
        - among the datasets that still have no started job.
        - if none, among the datasets that have the least started jobs:
          - in the limit of `max_jobs_per_namespace` jobs per namespace
          - ensuring that the unicity_id field is unique among the started jobs.

        Args:
            priority (`Priority`): The priority of the job.
            job_types_blocked: if not None, jobs of the given types are not considered.
            job_types_only: if not None, only jobs of the given types are considered.

        Raises:
            EmptyQueueError: if there is no waiting job in the queue that satisfies the restrictions above.

        Returns: the job
        """
        logging.debug(
            f"Getting next waiting job for priority {priority}, blocked types: {job_types_blocked}, only types:"
            f" {job_types_only}"
        )
        filters = {}
        if job_types_blocked:
            filters["type__nin"] = job_types_blocked
        if job_types_only:
            filters["type__in"] = job_types_only
        started_jobs = Job.objects(status=Status.STARTED, **filters)
        logging.debug(f"Number of started jobs: {started_jobs.count()}")
        started_job_namespaces = [job.namespace for job in started_jobs.only("namespace")]
        logging.debug(f"Started job namespaces: {started_job_namespaces}")

        next_waiting_job = (
            Job.objects(
                status=Status.WAITING, namespace__nin=set(started_job_namespaces), priority=priority, **filters
            )
            .order_by("+created_at")
            .only("type", "dataset", "revision", "config", "split", "priority")
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
                Job.objects(
                    status=Status.WAITING,
                    namespace__in=least_common_namespaces_group,
                    unicity_id__nin=started_unicity_ids,
                    priority=priority,
                    **filters,
                )
                .order_by("+created_at")
                .only("type", "dataset", "revision", "config", "split", "priority")
                .no_cache()
                .first()
            )
            if next_waiting_job is not None:
                return next_waiting_job
        raise EmptyQueueError(
            f"no job available with the priority (within the limit of {self.max_jobs_per_namespace} started jobs per"
            " namespace)"
        )

    def get_next_waiting_job(
        self, job_types_blocked: Optional[list[str]] = None, job_types_only: Optional[list[str]] = None
    ) -> Job:
        """Get the next job in the queue.

        Get the waiting job with the oldest creation date with the following criteria:
        - among the highest priority jobs,
        - among the datasets that still have no started job.
        - if none, among the datasets that have the least started jobs:
          - in the limit of `max_jobs_per_namespace` jobs per namespace
          - ensuring that the unicity_id field is unique among the started jobs.

        Args:
            job_types_blocked: if not None, jobs of the given types are not considered.
            job_types_only: if not None, only jobs of the given types are considered.

        Raises:
            EmptyQueueError: if there is no waiting job in the queue that satisfies the restrictions above.

        Returns: the job
        """
        for priority in [Priority.NORMAL, Priority.LOW]:
            with contextlib.suppress(EmptyQueueError):
                return self._get_next_waiting_job_for_priority(
                    priority=priority, job_types_blocked=job_types_blocked, job_types_only=job_types_only
                )
        raise EmptyQueueError(
            f"no job available (within the limit of {self.max_jobs_per_namespace} started jobs per namespace)"
        )

    def start_job(
        self, job_types_blocked: Optional[list[str]] = None, job_types_only: Optional[list[str]] = None
    ) -> JobInfo:
        """Start the next job in the queue.

        The job is moved from the waiting state to the started state.

        Args:
            job_types_blocked: if not None, jobs of the given types are not considered.
            job_types_only: if not None, only jobs of the given types are considered.

        Raises:
            EmptyQueueError: if there is no job in the queue, within the limit of the maximum number of started jobs
            for a dataset

        Returns: the job id, the type, the input arguments: dataset, revision, config and split
        """
        logging.debug(f"looking for a job to start, blocked types: {job_types_blocked}, only types: {job_types_only}")
        next_waiting_job = self.get_next_waiting_job(
            job_types_blocked=job_types_blocked, job_types_only=job_types_only
        )
        logging.debug(f"job found: {next_waiting_job}")
        # ^ can raise EmptyQueueError
        next_waiting_job.update(started_at=get_datetime(), status=Status.STARTED)
        if job_types_blocked and next_waiting_job.type in job_types_blocked:
            raise RuntimeError(
                f"The job type {next_waiting_job.type} is in the list of blocked job types {job_types_only}"
            )
        if job_types_only and next_waiting_job.type not in job_types_only:
            raise RuntimeError(
                f"The job type {next_waiting_job.type} is not in the list of allowed job types {job_types_only}"
            )
        return next_waiting_job.info()

    def get_job_with_id(self, job_id: str) -> Job:
        """Get the job for a given job id.

        Args:
            job_id (`str`, required): id of the job

        Returns: the requested job

        Raises:
            DoesNotExist: if the job does not exist
        """
        return Job.objects(pk=job_id).get()

    def get_job_type(self, job_id: str) -> str:
        """Get the job type for a given job id.

        Args:
            job_id (`str`, required): id of the job

        Returns: the job type

        Raises:
            DoesNotExist: if the job does not exist
        """
        job = self.get_job_with_id(job_id=job_id)
        return job.type

    def finish_job(self, job_id: str, is_success: bool) -> bool:
        """Finish a job in the queue.

        The job is moved from the started state to the success or error state.

        Args:
            job_id (`str`, required): id of the job
            is_success (`bool`, required): whether the job succeeded or not

        Returns:
            `bool`: whether the job existed, and had the expected format (STARTED status, non-empty started_at, empty
            finished_at) before finishing
        """
        result = True
        try:
            job = Job.objects(pk=job_id).get()
        except DoesNotExist:
            logging.error(f"job {job_id} does not exist. Aborting.")
            return False
        if job.status is not Status.STARTED:
            logging.warning(
                f"job {job.unicity_id} has a not the STARTED status ({job.status.value}). Force finishing anyway."
            )
            result = False
        if job.finished_at is not None:
            logging.warning(f"job {job.unicity_id} has a non-empty finished_at field. Force finishing anyway.")
            result = False
        if job.started_at is None:
            logging.warning(f"job {job.unicity_id} has an empty started_at field. Force finishing anyway.")
            result = False
        finished_status = Status.SUCCESS if is_success else Status.ERROR
        job.update(finished_at=get_datetime(), status=finished_status)
        return result

    def is_job_in_process(
        self, job_type: str, dataset: str, revision: str, config: Optional[str] = None, split: Optional[str] = None
    ) -> bool:
        """Check if a job is in process (waiting or started).

        Args:
            job_type (`str`, required): job type
            dataset (`str`, required): dataset name
            revision (`str`, required): dataset git revision
            config (`str`, optional): config name. Defaults to None.
            split (`str`, optional): split name. Defaults to None.

        Returns:
            `bool`: whether the job is in process (waiting or started)
        """
        return (
            Job.objects(
                type=job_type,
                dataset=dataset,
                revision=revision,
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
            self.upsert_job(
                job_type=job.type, dataset=job.dataset, revision=job.revision, config=job.config, split=job.split
            )

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

    def get_dataset_pending_jobs_for_type(self, dataset: str, job_type: str) -> List[JobDict]:
        """Get the pending jobs of a dataset for a given job type.

        Returns: an array of the pending jobs for the dataset and the given job type
        """
        return [
            d.to_dict()
            for d in Job.objects(
                type=job_type, dataset=dataset, status__in=[Status.WAITING.value, Status.STARTED.value]
            )
        ]

    def heartbeat(self, job_id: str) -> None:
        """Update the job `last_heartbeat` field with the current date.
        This is used to keep track of running jobs.
        If a job doesn't have recent heartbeats, it means it crashed at one point and is considered a zombie.
        """
        try:
            job = self.get_job_with_id(job_id)
        except DoesNotExist:
            logging.warning(f"Heartbeat skipped because job {job_id} doesn't exist in the queue.")
            return
        job.update(last_heartbeat=get_datetime())

    def get_zombies(self, max_seconds_without_heartbeat: int) -> List[JobInfo]:
        """Get the zombie jobs.
        It returns jobs without recent heartbeats, which means they crashed at one point and became zombies.
        Usually `max_seconds_without_heartbeat` is a factor of the time between two heartbeats.

        Returns: an array of the zombie job infos.
        """
        started_jobs = Job.objects(status=Status.STARTED)
        if max_seconds_without_heartbeat <= 0:
            return []
        zombies = [
            job
            for job in started_jobs
            if (
                job.last_heartbeat is not None
                and get_datetime()
                >= pytz.UTC.localize(job.last_heartbeat) + timedelta(seconds=max_seconds_without_heartbeat)
            )
            or (
                job.last_heartbeat is None
                and job.started_at is not None
                and get_datetime()
                >= pytz.UTC.localize(job.started_at) + timedelta(seconds=max_seconds_without_heartbeat)
            )
        ]
        return [zombie.info() for zombie in zombies]


# only for the tests
def _clean_queue_database() -> None:
    """Delete all the jobs in the database"""
    Job.drop_collection()  # type: ignore


# explicit re-export
__all__ = ["DoesNotExist"]
