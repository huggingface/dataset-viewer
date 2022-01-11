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
    dataset_name: str
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]


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
class Job(Document):
    meta = {"collection": "jobs", "db_alias": "queue"}
    dataset_name = StringField(required=True)
    created_at = DateTimeField(required=True)
    started_at = DateTimeField()
    finished_at = DateTimeField()

    def to_dict(self) -> JobDict:
        return {
            "dataset_name": self.dataset_name,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    objects = QuerySetManager["Job"]()


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


class InvalidJobId(Exception):
    pass


def add_job(dataset_name: str) -> None:
    try:
        # Check if a not-finished job already exists
        Job.objects(dataset_name=dataset_name, finished_at=None).get()
    except DoesNotExist:
        Job(dataset_name=dataset_name, created_at=datetime.utcnow()).save()
    # raises MultipleObjectsReturned if more than one entry -> should never occur, we let it raise


def get_waiting_jobs() -> QuerySet[Job]:
    return Job.objects(started_at=None)


def get_started_jobs() -> QuerySet[Job]:
    return Job.objects(started_at__exists=True, finished_at=None)


def get_finished_jobs() -> QuerySet[Job]:
    return Job.objects(finished_at__exists=True)


def get_job() -> Tuple[str, str]:
    job = get_waiting_jobs().order_by("+created_at").first()
    if job is None:
        raise EmptyQueue("no job available")
    if job.finished_at is not None:
        raise IncoherentState("a job with an empty start_at field should not have a finished_at field")
    job.update(started_at=datetime.utcnow())
    return str(job.id), job.dataset_name  # type: ignore


def finish_job(job_id: str) -> None:
    try:
        job = Job.objects(id=job_id, started_at__exists=True, finished_at=None).get()
    except DoesNotExist:
        raise JobNotFound("the job does not exist")
    except ValidationError:
        raise InvalidJobId("the job id is invalid")
    job.update(finished_at=datetime.utcnow())


def clean_database() -> None:
    Job.drop_collection()  # type: ignore


def force_finish_started_jobs() -> None:
    get_started_jobs().update(finished_at=datetime.utcnow())


# special reports


def get_jobs_count_with_status(status: str) -> int:
    if status == "waiting":
        return get_waiting_jobs().count()
    elif status == "started":
        return get_started_jobs().count()
    else:
        # done
        return get_finished_jobs().count()


def get_dump_by_status() -> DumpByStatus:
    return {
        "waiting": [d.to_dict() for d in get_waiting_jobs()],
        "started": [d.to_dict() for d in get_started_jobs()],
        "finished": [d.to_dict() for d in get_finished_jobs()],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
