# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import enum
import logging
import types
from typing import Generic, Type, TypeVar

from mongoengine import Document
from mongoengine.connection import get_db
from mongoengine.fields import BooleanField, DateTimeField, EnumField, StringField
from mongoengine.queryset.queryset import QuerySet

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationAddPriorityToJob(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        logging.info("Add the priority field, with the default value ('normal'), to all the jobs")
        db = get_db("queue")
        db["jobsBlue"].update_many({}, {"$set": {"priority": "normal"}})

    def down(self) -> None:
        logging.info("Remove the priority field from all the jobs")
        db = get_db("queue")
        db["jobsBlue"].update_many({}, {"$unset": {"priority": ""}})

    def validate(self) -> None:
        logging.info("Ensure that a random selection of jobs have the 'priority' field set to 'normal'")

        def custom_validation(doc: JobSnapshot) -> None:
            if doc.priority != Priority.NORMAL:
                raise ValueError("priority should be 'normal'")

        check_documents(DocCls=JobSnapshot, sample_size=10, custom_validation=custom_validation)
        if JobSnapshot.objects(priority=Priority.NORMAL).count() != JobSnapshot.objects.count():
            raise ValueError('All the objects should have the "priority" field, set to "normal"')


# --- JobSnapshot ---
# copied from libcommon.queue.Job, as a snapshot of when the migration was created
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


class JobSnapshot(Document):
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
        priority (`Priority`, optional): The priority of the job. Defaults to Priority.NORMAL.
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
            ("type", "dataset", "config", "split", "status", "force", "priority"),
            ("status", "type", "created_at", "namespace", "unicity_id", "priority"),
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

    objects = QuerySetManager["JobSnapshot"]()
