# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import datetime
from abc import ABC, abstractmethod

from libcommon.constants import (
    CACHE_COLLECTION_RESPONSES,
    CACHE_MONGOENGINE_ALIAS,
    METRICS_COLLECTION_CACHE_TOTAL_METRIC,
    METRICS_COLLECTION_JOB_TOTAL_METRIC,
    METRICS_MONGOENGINE_ALIAS,
    QUEUE_COLLECTION_JOBS,
    QUEUE_MONGOENGINE_ALIAS,
)


class IrreversibleMigrationError(Exception):
    pass


class Migration(ABC):
    def __init__(self, version: str, description: str):
        if version is None or description is None:
            raise ValueError("The version and the description are required.")
        try:
            datetime.datetime.strptime(version, "%Y%m%d%H%M%S")
        except Exception as e:
            raise ValueError("The version should be a string representing a date in the format YYYYMMDDHHMMSS") from e
        self.version = version
        self.description = description

    @abstractmethod
    def up(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def validate(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def down(self) -> None:
        raise IrreversibleMigrationError()


class QueueMigration(Migration):
    MONGOENGINE_ALIAS: str = QUEUE_MONGOENGINE_ALIAS
    COLLECTION_JOBS: str = QUEUE_COLLECTION_JOBS

    def __init__(self, job_type: str, version: str, description: str):
        self.job_type = job_type
        super().__init__(version=version, description=description)


class CacheMigration(Migration):
    MONGOENGINE_ALIAS: str = CACHE_MONGOENGINE_ALIAS
    COLLECTION_RESPONSES: str = CACHE_COLLECTION_RESPONSES

    def __init__(self, cache_kind: str, version: str, description: str):
        self.cache_kind = cache_kind
        super().__init__(version=version, description=description)


class MetricsMigration(Migration):
    MONGOENGINE_ALIAS: str = METRICS_MONGOENGINE_ALIAS
    COLLECTION_JOB_TOTAL_METRIC: str = METRICS_COLLECTION_JOB_TOTAL_METRIC
    COLLECTION_CACHE_TOTAL_METRIC: str = METRICS_COLLECTION_CACHE_TOTAL_METRIC

    def __init__(self, job_type: str, cache_kind: str, version: str, description: str):
        self.job_type = job_type
        self.cache_kind = cache_kind
        super().__init__(version=version, description=description)
