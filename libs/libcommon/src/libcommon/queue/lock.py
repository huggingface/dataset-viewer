# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import contextlib
import json
import logging
import time
import types
from collections.abc import Sequence
from enum import IntEnum
from types import TracebackType
from typing import Generic, Literal, Optional, TypeVar

from mongoengine import Document
from mongoengine.errors import NotUniqueError
from mongoengine.fields import DateTimeField, IntField, StringField
from mongoengine.queryset.queryset import QuerySet

from libcommon.constants import (
    LOCK_TTL_SECONDS_NO_OWNER,
    LOCK_TTL_SECONDS_TO_START_JOB,
    LOCK_TTL_SECONDS_TO_WRITE_ON_GIT_BRANCH,
    QUEUE_COLLECTION_LOCKS,
    QUEUE_MONGOENGINE_ALIAS,
)
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


# END monkey patching ### hack ###


class _TTL(IntEnum):
    LOCK_TTL_SECONDS_TO_START_JOB = LOCK_TTL_SECONDS_TO_START_JOB
    LOCK_TTL_SECONDS_TO_WRITE_ON_GIT_BRANCH = LOCK_TTL_SECONDS_TO_WRITE_ON_GIT_BRANCH


class Lock(Document):
    meta = {
        "collection": QUEUE_COLLECTION_LOCKS,
        "db_alias": QUEUE_MONGOENGINE_ALIAS,
        "indexes": [
            ("key", "owner"),
            {
                "name": "LOCK_TTL_SECONDS_NO_OWNER",
                "fields": ["updated_at"],
                "expireAfterSeconds": LOCK_TTL_SECONDS_NO_OWNER,
                "partialFilterExpression": {"owner": None},
            },
        ]
        + [
            {
                "name": ttl.name,
                "fields": ["updated_at"],
                "expireAfterSeconds": ttl,
                "partialFilterExpression": {"ttl": ttl},
            }
            for ttl in _TTL
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

    TTL = _TTL
    _default_sleeps = (0.05, 0.05, 0.05, 1, 1, 1, 5)

    def __init__(
        self, key: str, owner: str, sleeps: Sequence[float] = _default_sleeps, ttl: Optional[_TTL] = None
    ) -> None:
        self.key = key
        self.owner = owner
        self.sleeps = sleeps
        self.ttl = ttl
        if ttl is not None and ttl not in list(self.TTL):
            raise ValueError(f"The TTL value is not supported by the TTL index. It should be one of {list(self.TTL)}")

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
    def git_branch(
        cls,
        dataset: str,
        branch: str,
        owner: str,
        sleeps: Sequence[float] = _default_sleeps,
    ) -> "lock":
        """
        Lock a git branch of a dataset on the hub for read/write

        Args:
            dataset (`str`): the dataset repository
            branch (`str`): the branch to lock
            owner (`str`): the current job id that holds the lock
            sleeps (`Sequence[float]`, *optional*): the time in seconds to sleep between each attempt to acquire the lock
        """
        key = json.dumps({"dataset": dataset, "branch": branch})
        return cls(key=key, owner=owner, sleeps=sleeps, ttl=_TTL.LOCK_TTL_SECONDS_TO_WRITE_ON_GIT_BRANCH)


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


def release_lock(key: str) -> None:
    """
    Release the lock for a specific key

    Args:
        key (`str`): the lock key
    """
    Lock.objects(key=key).update(
        write_concern={"w": "majority", "fsync": True},
        read_concern={"level": "majority"},
        owner=None,
        updated_at=get_datetime(),
    )
