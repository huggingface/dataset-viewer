import time
import types
from datetime import datetime
from typing import Generic, List, Optional, Tuple, Type, TypedDict, TypeVar

from mongoengine import Document, DoesNotExist, connect
from mongoengine.errors import ValidationError
from mongoengine.fields import DateTimeField, StringField
from mongoengine.queryset.queryset import QuerySet

from datasets_preview_backend.config import MONGO_QUEUE_DATABASE, MONGO_URL

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
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]


class DatasetJobDict(JobDict):
    dataset_name: str


class SplitJobDict(JobDict):
    dataset_name: str
    config_name: str
    split_name: str


class DumpByStatus(TypedDict):
    waiting: List[JobDict]
    started: List[JobDict]
    finished: List[JobDict]
    created_at: str


def connect_to_queue() -> None:
    connect(MONGO_QUEUE_DATABASE, alias="queue", host=MONGO_URL)


# States:
# - waiting: started_at is None and finished_at is None: waiting jobs
# - started: started_at is not None and finished_at is None: started jobs
# - finished: started_at is not None and finished_at is None: started jobs
# For a given dataset_name, any number of finished jobs are allowed, but only 0 or 1
# job for the set of the other states
class DatasetJob(Document):
    meta = {"collection": "dataset_jobs", "db_alias": "queue"}
    dataset_name = StringField(required=True)
    created_at = DateTimeField(required=True)
    started_at = DateTimeField()
    finished_at = DateTimeField()

    def to_dict(self) -> DatasetJobDict:
        return {
            "dataset_name": self.dataset_name,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    objects = QuerySetManager["DatasetJob"]()


class SplitJob(Document):
    meta = {"collection": "split_jobs", "db_alias": "queue"}
    dataset_name = StringField(required=True)
    config_name = StringField(required=True)
    split_name = StringField(required=True)
    created_at = DateTimeField(required=True)
    started_at = DateTimeField()
    finished_at = DateTimeField()

    def to_dict(self) -> SplitJobDict:
        return {
            "dataset_name": self.dataset_name,
            "config_name": self.config_name,
            "split_name": self.split_name,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    objects = QuerySetManager["SplitJob"]()


AnyJob = TypeVar("AnyJob", DatasetJob, SplitJob)  # Must be DatasetJob or SplitJob


# TODO: add priority (webhook: 5, warming: 3, refresh: 1)
# TODO: add status (valid/error/stalled) to the finished jobs
# TODO: limit the size of the queue? remove the oldest if room is needed?
# TODO: how to avoid deadlocks (a worker has taken the job, but never finished)? stalled, hours

# enqueue
# dequeue
# peek
# isfull
# isempty


class EmptyQueue(Exception):
    pass


class JobNotFound(Exception):
    pass


class IncoherentState(Exception):
    pass


def add_job(existing_jobs: QuerySet[AnyJob], new_job: AnyJob):
    try:
        # Check if a non-finished job already exists
        existing_jobs.filter(finished_at=None).get()
    except DoesNotExist:
        new_job.save()
    # raises MultipleObjectsReturned if more than one entry -> should never occur, we let it raise


def add_dataset_job(dataset_name: str) -> None:
    add_job(
        DatasetJob.objects(dataset_name=dataset_name),
        DatasetJob(dataset_name=dataset_name, created_at=datetime.utcnow()),
    )


def add_split_job(dataset_name: str, config_name: str, split_name: str) -> None:
    add_job(
        SplitJob.objects(dataset_name=dataset_name, config_name=config_name, split_name=split_name),
        SplitJob(
            dataset_name=dataset_name, config_name=config_name, split_name=split_name, created_at=datetime.utcnow()
        ),
    )


def get_waiting(jobs: QuerySet[AnyJob]) -> QuerySet[AnyJob]:
    return jobs(started_at__exists=False, finished_at__exists=False)


def get_started(jobs: QuerySet[AnyJob]) -> QuerySet[AnyJob]:
    return jobs(started_at__exists=True, finished_at__exists=False)


def get_finished(jobs: QuerySet[AnyJob]) -> QuerySet[AnyJob]:
    return jobs(finished_at__exists=True)


def start_job(jobs: QuerySet[AnyJob]) -> AnyJob:
    job = get_waiting(jobs).order_by("+created_at").first()
    if job is None:
        raise EmptyQueue("no job available")
    if job.finished_at is not None:
        raise IncoherentState("a job with an empty start_at field should not have a finished_at field")
    job.update(started_at=datetime.utcnow())
    return job


def get_dataset_job() -> Tuple[str, str]:
    job = start_job(DatasetJob.objects)
    return str(job.pk), job.dataset_name
    # ^ job.pk is the id. job.id is not recognized by mypy


def get_split_job() -> Tuple[str, str, str, str]:
    job = start_job(SplitJob.objects)
    return str(job.pk), job.dataset_name, job.config_name, job.split_name
    # ^ job.pk is the id. job.id is not recognized by mypy


def finish_jobs(jobs: QuerySet[AnyJob], job_id=str) -> None:
    try:
        job = jobs(pk=job_id, started_at__exists=True, finished_at=None).get()
    except (DoesNotExist, ValidationError):
        raise JobNotFound("the job does not exist")
    job.update(finished_at=datetime.utcnow())


def finish_started_jobs(jobs: QuerySet[AnyJob]) -> None:
    get_started(jobs).update(finished_at=datetime.utcnow())


def finish_waiting_jobs(jobs: QuerySet[AnyJob]) -> None:
    get_waiting(jobs).update(finished_at=datetime.utcnow())


def finish_dataset_job(job_id: str) -> None:
    finish_jobs(DatasetJob.objects, job_id)


def finish_split_job(job_id: str) -> None:
    finish_jobs(SplitJob.objects, job_id)


def clean_database() -> None:
    DatasetJob.drop_collection()  # type: ignore
    SplitJob.drop_collection()  # type: ignore


def finish_started_dataset_jobs() -> None:
    finish_started_jobs(DatasetJob.objects)


def finish_started_split_jobs() -> None:
    finish_started_jobs(SplitJob.objects)


def finish_waiting_dataset_jobs() -> None:
    finish_waiting_jobs(DatasetJob.objects)


def finish_waiting_split_jobs() -> None:
    finish_waiting_jobs(SplitJob.objects)


# special reports


def get_jobs_count_with_status(jobs: QuerySet[AnyJob], status: str) -> int:
    if status == "waiting":
        return get_waiting(jobs).count()
    elif status == "started":
        return get_started(jobs).count()
    else:
        # done
        return get_finished(jobs).count()


def get_dataset_jobs_count_with_status(status: str) -> int:
    return get_jobs_count_with_status(DatasetJob.objects, status)


def get_split_jobs_count_with_status(status: str) -> int:
    return get_jobs_count_with_status(SplitJob.objects, status)


def get_dump_by_status(jobs: QuerySet[AnyJob]) -> DumpByStatus:
    return {
        "waiting": [d.to_dict() for d in get_waiting(jobs)],
        "started": [d.to_dict() for d in get_started(jobs)],
        "finished": [d.to_dict() for d in get_finished(jobs)],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def get_dataset_dump_by_status() -> DumpByStatus:
    return get_dump_by_status(DatasetJob.objects)


def get_split_dump_by_status() -> DumpByStatus:
    return get_dump_by_status(SplitJob.objects)
