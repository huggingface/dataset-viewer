# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import contextlib
import json
import logging
import time
import types
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timedelta
from itertools import groupby
from operator import itemgetter
from types import TracebackType
from typing import Generic, Literal, Optional, TypedDict, TypeVar
from uuid import uuid4

import pandas as pd
import pytz
from bson import ObjectId
from mongoengine import Document
from mongoengine.errors import DoesNotExist, NotUniqueError
from mongoengine.fields import (
    DateTimeField,
    EnumField,
    IntField,
    ObjectIdField,
    StringField,
)
from mongoengine.queryset.queryset import QuerySet

from libcommon.constants import (
    DEFAULT_DIFFICULTY_MAX,
    DEFAULT_DIFFICULTY_MIN,
    LOCK_TTL_SECONDS,
    QUEUE_COLLECTION_JOBS,
    QUEUE_COLLECTION_LOCKS,
    QUEUE_METRICS_COLLECTION,
    QUEUE_MONGOENGINE_ALIAS,
    QUEUE_TTL_SECONDS,
)
from libcommon.utils import (
    FlatJobInfo,
    JobInfo,
    Priority,
    Status,
    get_datetime,
    inputs_to_string,
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


class StartedJobError(Exception):
    pass


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
    difficulty: int
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
    waiting: list[JobDict]
    started: list[JobDict]


class EmptyQueueError(Exception):
    pass


class JobDoesNotExistError(DoesNotExist):
    pass


class AlreadyStartedJobError(Exception):
    pass


class LockTimeoutError(Exception):
    pass


class NoWaitingJobError(Exception):
    pass


class JobQueryFilters(TypedDict, total=False):
    type__nin: list[str]
    type__in: list[str]
    difficulty__gte: int
    difficulty__lte: int


# States:
# - waiting: started_at is None and finished_at is None: waiting jobs
# - started: started_at is not None and finished_at is None: started jobs
# - finished: started_at is not None and finished_at is not None: finished jobs
# For a given set of arguments, only one job is allowed in the started state. No
# restriction for the other states
class JobDocument(Document):
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
        priority (`Priority`, optional): The priority of the job. Defaults to Priority.LOW.
        status (`Status`, optional): The status of the job. Defaults to Status.WAITING.
        difficulty (`int`): The difficulty of the job: 0=easy, 100=hard as a convention.
        created_at (`datetime`): The creation date of the job.
        started_at (`datetime`, optional): When the job has started.
        finished_at (`datetime`, optional): When the job has finished.
        last_heartbeat (`datetime`, optional): Last time the running job got a heartbeat from the worker.
    """

    meta = {
        "collection": QUEUE_COLLECTION_JOBS,
        "db_alias": QUEUE_MONGOENGINE_ALIAS,
        "indexes": [
            ("type", "dataset", "status"),
            ("type", "dataset", "revision", "config", "split", "status", "priority"),
            ("priority", "status", "created_at", "namespace"),
            ("priority", "status", "type", "namespace", "unicity_id", "created_at", "-difficulty"),
            ("status", "type"),
            ("unicity_id", "-created_at", "status"),
            {
                "fields": ["finished_at"],
                "expireAfterSeconds": QUEUE_TTL_SECONDS,
                "partialFilterExpression": {"status": {"$in": [Status.SUCCESS, Status.ERROR, Status.CANCELLED]}},
            },
        ],
    }
    type = StringField(required=True)
    dataset = StringField(required=True)
    revision = StringField(required=True)
    config = StringField()
    split = StringField()
    unicity_id = StringField(required=True)
    namespace = StringField(required=True)
    priority = EnumField(Priority, default=Priority.LOW)
    status = EnumField(Status, default=Status.WAITING)
    difficulty = IntField(required=True)
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
            "difficulty": self.difficulty,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_heartbeat": self.last_heartbeat,
        }

    objects = QuerySetManager["JobDocument"]()

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
                "difficulty": self.difficulty,
            }
        )

    @classmethod
    def get(cls, job_id: str) -> "JobDocument":
        try:
            return cls.objects(pk=job_id).get()
        except DoesNotExist as e:
            raise JobDoesNotExistError(f"Job does not exist: {job_id=}") from e

    def flat_info(self) -> FlatJobInfo:
        return FlatJobInfo(
            {
                "job_id": str(self.pk),  # job.pk is the id. job.id is not recognized by mypy
                "type": self.type,
                "dataset": self.dataset,
                "revision": self.revision,
                "config": self.config,
                "split": self.split,
                "priority": self.priority.value,
                "status": self.status.value,
                "difficulty": self.difficulty,
                "created_at": self.created_at,
            }
        )


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
        "collection": QUEUE_METRICS_COLLECTION,
        "db_alias": QUEUE_MONGOENGINE_ALIAS,
        "indexes": [("job_type", "status")],
    }
    objects = QuerySetManager["JobTotalMetricDocument"]()


def _update_metrics(job_type: str, status: str, increase_by: int) -> None:
    JobTotalMetricDocument.objects(job_type=job_type, status=status).update(
        upsert=True,
        write_concern={"w": "majority", "fsync": True},
        read_concern={"level": "majority"},
        inc__total=increase_by,
    )


def increase_metric(job_type: str, status: str) -> None:
    _update_metrics(job_type=job_type, status=status, increase_by=DEFAULT_INCREASE_AMOUNT)


def decrease_metric(job_type: str, status: str) -> None:
    _update_metrics(job_type=job_type, status=status, increase_by=DEFAULT_DECREASE_AMOUNT)


def update_metrics_for_type(job_type: str, previous_status: str, new_status: str) -> None:
    if job_type is not None:
        decrease_metric(job_type=job_type, status=previous_status)
        increase_metric(job_type=job_type, status=new_status)


class Lock(Document):
    meta = {
        "collection": QUEUE_COLLECTION_LOCKS,
        "db_alias": QUEUE_MONGOENGINE_ALIAS,
        "indexes": [
            ("key", "owner"),
            {
                "fields": ["updated_at"],
                "expireAfterSeconds": LOCK_TTL_SECONDS,
                "partialFilterExpression": {"$or": [{"owner": None}, {"ttl": LOCK_TTL_SECONDS}]},
            },
        ],
    }

    key = StringField(primary_key=True)
    owner = StringField()
    ttl = IntField()
    job_id = StringField()  # deprecated

    created_at = DateTimeField()
    updated_at = DateTimeField()

    objects = QuerySetManager["Lock"]()


class lock(contextlib.AbstractContextManager["lock"]):
    """
    Provides a simple way of inter-applications communication using a MongoDB lock.

    An example usage is to another worker of your application that a resource
    or working directory is currently used in a job.

    Example of usage:

    ```python
    key = json.dumps({"type": job.type, "dataset": job.dataset})
    with lock(key=key, owner=job.pk):
        ...
    ```

    Or using a try/except:

    ```python
    try:
        key = json.dumps({"type": job.type, "dataset": job.dataset})
        lock(key=key, owner=job.pk).acquire()
    except TimeoutError:
        ...
    ```
    """

    _default_sleeps = (0.05, 0.05, 0.05, 1, 1, 1, 5)

    def __init__(
        self, key: str, owner: str, sleeps: Sequence[float] = _default_sleeps, ttl: Optional[int] = None
    ) -> None:
        self.key = key
        self.owner = owner
        self.sleeps = sleeps
        self.ttl = ttl
        if ttl is not None and ttl != LOCK_TTL_SECONDS:
            raise ValueError(f"Only TTL of LOCK_TTL_SECONDS={LOCK_TTL_SECONDS} is supported by the TTL index.")

    def acquire(self) -> None:
        for sleep in self.sleeps:
            try:
                Lock.objects(key=self.key, owner__in=[None, self.owner]).update(
                    upsert=True,
                    write_concern={"w": "majority", "fsync": True},
                    read_concern={"level": "majority"},
                    owner=self.owner,
                    updated_at=get_datetime(),
                    ttl=self.ttl,
                )
                return
            except NotUniqueError:
                logging.debug(f"Sleep {sleep}s to acquire lock '{self.key}' for owner='{self.owner}'")
                time.sleep(sleep)
        raise TimeoutError("lock couldn't be acquired")

    def release(self) -> None:
        Lock.objects(key=self.key, owner=self.owner).update(
            write_concern={"w": "majority", "fsync": True},
            read_concern={"level": "majority"},
            owner=None,
            updated_at=get_datetime(),
        )

    def __enter__(self) -> "lock":
        self.acquire()
        return self

    def __exit__(
        self, exctype: Optional[type[BaseException]], excinst: Optional[BaseException], exctb: Optional[TracebackType]
    ) -> Literal[False]:
        self.release()
        return False

    @classmethod
    def git_branch(cls, dataset: str, branch: str, owner: str, sleeps: Sequence[float] = _default_sleeps) -> "lock":
        """
        Lock a git branch of a dataset on the hub for read/write

        Args:
            dataset (`str`): the dataset repository
            branch (`str`): the branch to lock
            owner (`str`): the current job id that holds the lock
            sleeps (`Sequence[float]`): the time in seconds to sleep between each attempt to acquire the lock
        """
        key = json.dumps({"dataset": dataset, "branch": branch})
        return cls(key=key, owner=owner, sleeps=sleeps)


def release_locks(owner: str) -> None:
    """
    Release all locks owned by the given owner

    Args:
        owner (`str`): the current owner that holds the locks
    """
    Lock.objects(owner=owner).update(
        write_concern={"w": "majority", "fsync": True},
        read_concern={"level": "majority"},
        owner=None,
        updated_at=get_datetime(),
    )


class Queue:
    """A queue manages jobs.

    Note that creating a Queue object does not create the queue in the database. It's a view that allows to manipulate
    the jobs. You can create multiple Queue objects, it has no effect on the database.

    It's a FIFO queue, with the following properties:
    - a job is identified by its input arguments: unicity_id (type, dataset, config and split, NOT revision)
    - a job can be in one of the following states: waiting, started, success, error, cancelled
    - a job can be in the queue only once (unicity_id) in the "started" state
    - a job can be in the queue multiple times in the other states
    - a job has a priority (two levels: NORMAL and LOW)
    - a job has a difficulty (from 0: easy to 100: hard, as a convention)
    - the queue is ordered by priority then by the creation date of the jobs
    - datasets and users that already have started jobs are de-prioritized (using namespace)
    """

    def add_job(
        self,
        job_type: str,
        dataset: str,
        revision: str,
        difficulty: int,
        config: Optional[str] = None,
        split: Optional[str] = None,
        priority: Priority = Priority.LOW,
    ) -> JobDocument:
        """Add a job to the queue in the waiting state.

        Note that the same "unicity_id" can have multiple jobs in the waiting state, with the same or different
        revisions and or priorities.

        Args:
            job_type (`str`): The type of the job
            dataset (`str`): The dataset on which to apply the job.
            revision (`str`): The git revision of the dataset.
            difficulty (`int`): The difficulty of the job.
            config (`str`, optional): The config on which to apply the job.
            split (`str`, optional): The config on which to apply the job.
            priority (`Priority`, optional): The priority of the job. Defaults to Priority.LOW.

        Returns: the job
        """
        increase_metric(job_type=job_type, status=Status.WAITING)
        return JobDocument(
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
            difficulty=difficulty,
        ).save()

    def create_jobs(self, job_infos: list[JobInfo]) -> int:
        """Creates jobs in the queue.

        They are created in the waiting state.

        Args:
            job_infos (`list[JobInfo]`): The jobs to be created.

        Returns:
            `int`: The number of created jobs. 0 if we had an exception.
        """
        try:
            jobs = [
                JobDocument(
                    type=job_info["type"],
                    dataset=job_info["params"]["dataset"],
                    revision=job_info["params"]["revision"],
                    config=job_info["params"]["config"],
                    split=job_info["params"]["split"],
                    unicity_id=inputs_to_string(
                        dataset=job_info["params"]["dataset"],
                        config=job_info["params"]["config"],
                        split=job_info["params"]["split"],
                        prefix=job_info["type"],
                    ),
                    namespace=job_info["params"]["dataset"].split("/")[0],
                    priority=job_info["priority"],
                    created_at=get_datetime(),
                    status=Status.WAITING,
                    difficulty=job_info["difficulty"],
                )
                for job_info in job_infos
            ]
            for job in jobs:
                increase_metric(job_type=job.type, status=Status.WAITING)
            job_ids = JobDocument.objects.insert(jobs, load_bulk=False)
            return len(job_ids)
        except Exception:
            return 0

    def cancel_jobs_by_job_id(self, job_ids: list[str]) -> int:
        """Cancel jobs from the queue.

        If the job ids are not valid, they are ignored.

        Args:
            job_ids (`list[str]`): The list of job ids to cancel.

        Returns:
            `int`: The number of canceled jobs
        """
        try:
            existing = JobDocument.objects(pk__in=job_ids)
            previous_status = [(job.type, job.status) for job in existing.all()]
            existing.update(finished_at=get_datetime(), status=Status.CANCELLED)
            for job_type, status in previous_status:
                update_metrics_for_type(job_type=job_type, previous_status=status, new_status=Status.CANCELLED)
            return existing.count()
        except Exception:
            return 0

    def _get_next_waiting_job_for_priority(
        self,
        priority: Priority,
        difficulty_min: Optional[int] = None,
        difficulty_max: Optional[int] = None,
        job_types_blocked: Optional[list[str]] = None,
        job_types_only: Optional[list[str]] = None,
    ) -> JobDocument:
        """Get the next job in the queue for a given priority.

        For a given priority, get the waiting job with the oldest creation date:
        - among the datasets that still have no started job.
        - if none, among the datasets that have the least started jobs:
          - ensuring that the unicity_id field is unique among the started jobs.

        Args:
            priority (`Priority`): The priority of the job.
            difficulty_min: if not None, only jobs with a difficulty greater or equal to this value are considered.
            difficulty_max: if not None, only jobs with a difficulty lower or equal to this value are considered.
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
        filters: JobQueryFilters = {}
        if job_types_blocked:
            filters["type__nin"] = job_types_blocked
        if job_types_only:
            filters["type__in"] = job_types_only
        if difficulty_min is not None and difficulty_min > DEFAULT_DIFFICULTY_MIN:
            filters["difficulty__gte"] = difficulty_min
        if difficulty_max is not None and difficulty_max < DEFAULT_DIFFICULTY_MAX:
            filters["difficulty__lte"] = difficulty_max
        started_jobs = JobDocument.objects(status=Status.STARTED, **filters)
        logging.debug(f"Number of started jobs: {started_jobs.count()}")
        started_job_namespaces = [job.namespace for job in started_jobs.only("namespace")]
        logging.debug(f"Started job namespaces: {started_job_namespaces}")

        next_waiting_job = (
            JobDocument.objects(
                status=Status.WAITING, namespace__nin=set(started_job_namespaces), priority=priority, **filters
            )
            .order_by("+created_at")
            .only("type", "dataset", "revision", "config", "split", "priority", "unicity_id")
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
        # - exclude the waiting jobs which unicity_id is already in a started job
        # and, among the remaining waiting jobs, let's:
        # - select the oldest waiting job for the namespace with the least number of started jobs
        started_unicity_ids = {job.unicity_id for job in started_jobs.only("unicity_id")}
        descending_frequency_namespace_counts = [
            [namespace, count] for namespace, count in Counter(started_job_namespaces).most_common()
        ]
        logging.debug(f"Descending frequency namespace counts: {descending_frequency_namespace_counts}")
        descending_frequency_namespace_groups = [
            [item[0] for item in data] for (_, data) in groupby(descending_frequency_namespace_counts, itemgetter(1))
        ]
        # maybe we could get rid of this loop
        while descending_frequency_namespace_groups:
            least_common_namespaces_group = descending_frequency_namespace_groups.pop()
            logging.debug(f"Least common namespaces group: {least_common_namespaces_group}")
            next_waiting_job = (
                JobDocument.objects(
                    status=Status.WAITING,
                    namespace__in=least_common_namespaces_group,
                    unicity_id__nin=started_unicity_ids,
                    priority=priority,
                    **filters,
                )
                .order_by("+created_at")
                .only("type", "dataset", "revision", "config", "split", "priority", "unicity_id")
                .no_cache()
                .first()
            )
            if next_waiting_job is not None:
                return next_waiting_job
        raise EmptyQueueError("no job available with the priority")

    def get_next_waiting_job(
        self,
        difficulty_min: Optional[int] = None,
        difficulty_max: Optional[int] = None,
        job_types_blocked: Optional[list[str]] = None,
        job_types_only: Optional[list[str]] = None,
    ) -> JobDocument:
        """Get the next job in the queue.

        Get the waiting job with the oldest creation date with the following criteria:
        - among the highest priority jobs,
        - among the datasets that still have no started job.
        - if none, among the datasets that have the least started jobs:
          - ensuring that the unicity_id field is unique among the started jobs.

        Args:
            difficulty_min: if not None, only jobs with a difficulty greater or equal to this value are considered.
            difficulty_max: if not None, only jobs with a difficulty lower or equal to this value are considered.
            job_types_blocked: if not None, jobs of the given types are not considered.
            job_types_only: if not None, only jobs of the given types are considered.

        Raises:
            EmptyQueueError: if there is no waiting job in the queue that satisfies the restrictions above.

        Returns: the job
        """
        for priority in [Priority.NORMAL, Priority.LOW]:
            with contextlib.suppress(EmptyQueueError):
                return self._get_next_waiting_job_for_priority(
                    priority=priority,
                    job_types_blocked=job_types_blocked,
                    job_types_only=job_types_only,
                    difficulty_min=difficulty_min,
                    difficulty_max=difficulty_max,
                )
        raise EmptyQueueError("no job available")

    def _start_newest_job_and_cancel_others(self, job: JobDocument) -> JobDocument:
        """Start a job (the newest one for unicity_id) and cancel the other ones.

        A lock is used to ensure that the job is not started by another worker.

        Args:
            job: the job to start

        Returns:
            the started job

        Raises:
            AlreadyStartedJobError: if a started job already exist for the same unicity_id.
            LockTimeoutError: if the lock could not be acquired after 20 retries.
        """
        # could be a method of Job
        RETRIES = 20
        # uuid is used to differentiate between workers
        # otherwise another worker might acquire the lock
        lock_owner = str(uuid4())
        try:
            # retry for 2 seconds
            with lock(key=job.unicity_id, owner=lock_owner, sleeps=[0.1] * RETRIES, ttl=LOCK_TTL_SECONDS):
                # get all the pending jobs for the same unicity_id
                waiting_jobs = JobDocument.objects(
                    unicity_id=job.unicity_id, status__in=[Status.WAITING, Status.STARTED]
                ).order_by("-created_at")
                datetime = get_datetime()
                # raise if any job has already been started for unicity_id
                num_started_jobs = waiting_jobs(status=Status.STARTED).count()
                if num_started_jobs > 0:
                    if num_started_jobs > 1:
                        logging.critical(f"job {job.unicity_id} has been started {num_started_jobs} times. Max is 1.")
                    raise AlreadyStartedJobError(f"job {job.unicity_id} has been started by another worker")
                # get the most recent one
                first_job = waiting_jobs.first()
                if not first_job:
                    raise NoWaitingJobError(f"no waiting job could be found for {job.unicity_id}")
                # start it
                if not JobDocument.objects(pk=str(first_job.pk), status=Status.WAITING).update(
                    started_at=datetime,
                    status=Status.STARTED,
                    write_concern={"w": "majority", "fsync": True},
                    read_concern={"level": "majority"},
                ):
                    raise AlreadyStartedJobError(f"job {job.unicity_id} has been started by another worker")
                update_metrics_for_type(
                    job_type=first_job.type, previous_status=Status.WAITING, new_status=Status.STARTED
                )
                # and cancel the other ones, if any
                waiting_jobs(status=Status.WAITING).update(
                    finished_at=datetime,
                    status=Status.CANCELLED,
                    write_concern={"w": "majority", "fsync": True},
                    read_concern={"level": "majority"},
                )
                for waiting_job in waiting_jobs(status=Status.WAITING):
                    update_metrics_for_type(
                        job_type=waiting_job.type, previous_status=Status.WAITING, new_status=Status.CANCELLED
                    )
                return first_job.reload()
        except TimeoutError as err:
            raise LockTimeoutError(
                f"could not acquire the lock for job {job.unicity_id} after {RETRIES} retries."
            ) from err

    def start_job(
        self,
        difficulty_min: Optional[int] = None,
        difficulty_max: Optional[int] = None,
        job_types_blocked: Optional[list[str]] = None,
        job_types_only: Optional[list[str]] = None,
    ) -> JobInfo:
        """Start the next job in the queue.

        The job is moved from the waiting state to the started state. A lock is used to ensure that only one worker
        can start a job at a time.

        Args:
            difficulty_min: if not None, only jobs with a difficulty greater or equal to this value are considered.
            difficulty_max: if not None, only jobs with a difficulty lower or equal to this value are considered.
            job_types_blocked: if not None, jobs of the given types are not considered.
            job_types_only: if not None, only jobs of the given types are considered.

        Raises:
            EmptyQueueError: if there is no job in the queue, within the limit of the maximum number of started jobs
            for a dataset
            AlreadyStartedJobError: if a started job already exist for the same unicity_id
            LockTimeoutError: if the lock cannot be acquired

        Returns: the job id, the type, the input arguments: dataset, revision, config and split
        """

        logging.debug(f"looking for a job to start, blocked types: {job_types_blocked}, only types: {job_types_only}")
        next_waiting_job = self.get_next_waiting_job(
            job_types_blocked=job_types_blocked,
            job_types_only=job_types_only,
            difficulty_min=difficulty_min,
            difficulty_max=difficulty_max,
        )
        logging.debug(f"job found: {next_waiting_job}")
        # ^ can raise EmptyQueueError
        if job_types_blocked and next_waiting_job.type in job_types_blocked:
            raise RuntimeError(
                f"The job type {next_waiting_job.type} is in the list of blocked job types {job_types_only}"
            )
        if job_types_only and next_waiting_job.type not in job_types_only:
            raise RuntimeError(
                f"The job type {next_waiting_job.type} is not in the list of allowed job types {job_types_only}"
            )
        started_job = self._start_newest_job_and_cancel_others(job=next_waiting_job)
        return started_job.info()

    def get_job_with_id(self, job_id: str) -> JobDocument:
        """Get the job for a given job id.

        Args:
            job_id (`str`, required): id of the job

        Returns: the requested job

        Raises:
            DoesNotExist: if the job does not exist
        """
        return JobDocument.objects(pk=job_id).get()

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

    def _get_started_job(self, job_id: str) -> JobDocument:
        """Get a started job, and raise if it's not in the correct format
          (does not exist, not started, incorrect values for finished_at or started_at).

        Args:
            job_id (`str`, required): id of the job

        Returns:
            `Job`: the started job
        """
        job = JobDocument.objects(pk=job_id).get()
        if job.status is not Status.STARTED:
            raise StartedJobError(f"job {job.unicity_id} has a not the STARTED status ({job.status.value}).")
        if job.finished_at is not None:
            raise StartedJobError(f"job {job.unicity_id} has a non-empty finished_at field.")
        if job.started_at is None:
            raise StartedJobError(f"job {job.unicity_id} has an empty started_at field.")
        return job

    def is_job_started(self, job_id: str) -> bool:
        """Check if a job is started, with the correct values for finished_at and started_at.

        Args:
            job_id (`str`, required): id of the job

        Returns:
            `bool`: whether the job exists, is started, and had the expected format (STARTED status, non-empty
              started_at, empty finished_at)
        """
        try:
            self._get_started_job(job_id=job_id)
        except DoesNotExist:
            logging.error(f"job {job_id} does not exist.")
            return False
        except StartedJobError as e:
            logging.debug(f"job {job_id} has not the expected format for a started job: {e}")
            return False
        return True

    def finish_job(self, job_id: str, is_success: bool) -> bool:
        """Finish a job in the queue.

        The job is moved from the started state to the success or error state. The existing locks are released.

        Args:
            job_id (`str`, required): id of the job
            is_success (`bool`, required): whether the job succeeded or not

        Returns:
            `bool`: whether the job existed, and had the expected format (STARTED status, non-empty started_at, empty
            finished_at) before finishing
        """
        try:
            job = self._get_started_job(job_id=job_id)
        except DoesNotExist:
            logging.error(f"job {job_id} does not exist. Aborting.")
            return False
        except StartedJobError as e:
            logging.error(f"job {job_id} has not the expected format for a started job. Aborting: {e}")
            return False
        finished_status = Status.SUCCESS if is_success else Status.ERROR
        previous_status = job.status
        job.update(finished_at=get_datetime(), status=finished_status)
        update_metrics_for_type(job_type=job.type, previous_status=previous_status, new_status=finished_status)
        release_locks(owner=job_id)
        return True

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
            JobDocument.objects(
                type=job_type,
                dataset=dataset,
                revision=revision,
                config=config,
                split=split,
                status__in=[Status.WAITING, Status.STARTED],
            ).count()
            > 0
        )

    def _get_df(self, jobs: list[FlatJobInfo]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "job_id": pd.Series([job["job_id"] for job in jobs], dtype="str"),
                "type": pd.Series([job["type"] for job in jobs], dtype="category"),
                "dataset": pd.Series([job["dataset"] for job in jobs], dtype="str"),
                "revision": pd.Series([job["revision"] for job in jobs], dtype="str"),
                "config": pd.Series([job["config"] for job in jobs], dtype="str"),
                "split": pd.Series([job["split"] for job in jobs], dtype="str"),
                "priority": pd.Categorical(
                    [job["priority"] for job in jobs],
                    ordered=True,
                    categories=[Priority.LOW.value, Priority.NORMAL.value],
                ),
                "status": pd.Categorical(
                    [job["status"] for job in jobs],
                    ordered=True,
                    categories=[
                        Status.WAITING.value,
                        Status.STARTED.value,
                        Status.SUCCESS.value,
                        Status.ERROR.value,
                        Status.CANCELLED.value,
                    ],
                ),
                "created_at": pd.Series([job["created_at"] for job in jobs], dtype="datetime64[ns]"),
            }
        )
        # ^ does not seem optimal at all, but I get the types right

    def get_pending_jobs_df(self, dataset: str, job_types: Optional[list[str]] = None) -> pd.DataFrame:
        filters = {}
        if job_types:
            filters["type__in"] = job_types
        return self._get_df(
            [
                job.flat_info()
                for job in JobDocument.objects(status__in=[Status.WAITING, Status.STARTED], **filters, dataset=dataset)
            ]
        )

    def has_pending_jobs(self, dataset: str, job_types: Optional[list[str]] = None) -> bool:
        filters = {}
        if job_types:
            filters["type__in"] = job_types
        return JobDocument.objects(status__in=[Status.WAITING, Status.STARTED], **filters, dataset=dataset).count() > 0

    # special reports
    def count_jobs(self, status: Status, job_type: str) -> int:
        """Count the number of jobs with a given status and the given type.

        Args:
            status (`Status`, required): status of the jobs
            job_type (`str`, required): job type

        Returns: the number of jobs with the given status and the given type.
        """
        return JobDocument.objects(type=job_type, status=status.value).count()

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

    def get_dump_with_status(self, status: Status, job_type: str) -> list[JobDict]:
        """Get the dump of the jobs with a given status and a given type.

        Args:
            status (`Status`, required): status of the jobs
            job_type (`str`, required): job type

        Returns: a list of jobs with the given status and the given type
        """
        return [d.to_dict() for d in JobDocument.objects(status=status.value, type=job_type)]

    def get_dump_by_pending_status(self, job_type: str) -> DumpByPendingStatus:
        """Get the dump of the jobs by pending status for a given job type.

        Returns: a dictionary with the dump of the jobs for each pending status
        """
        return {
            "waiting": self.get_dump_with_status(job_type=job_type, status=Status.WAITING),
            "started": self.get_dump_with_status(job_type=job_type, status=Status.STARTED),
        }

    def get_dataset_pending_jobs_for_type(self, dataset: str, job_type: str) -> list[JobDict]:
        """Get the pending jobs of a dataset for a given job type.

        Returns: an array of the pending jobs for the dataset and the given job type
        """
        return [
            d.to_dict()
            for d in JobDocument.objects(
                status__in=[Status.WAITING.value, Status.STARTED.value], type=job_type, dataset=dataset
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
        # no need to update metrics since it is just the last_heartbeat
        job.update(last_heartbeat=get_datetime())

    def get_zombies(self, max_seconds_without_heartbeat: float) -> list[JobInfo]:
        """Get the zombie jobs.
        It returns jobs without recent heartbeats, which means they crashed at one point and became zombies.
        Usually `max_seconds_without_heartbeat` is a factor of the time between two heartbeats.

        Returns: an array of the zombie job infos.
        """
        started_jobs = JobDocument.objects(status=Status.STARTED)
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
    JobDocument.drop_collection()  # type: ignore
    JobTotalMetricDocument.drop_collection()  # type: ignore
    Lock.drop_collection()  # type: ignore
