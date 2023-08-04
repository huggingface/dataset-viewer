# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field
from typing import List, Optional

from environs import Env
from libcommon.config import (
    CacheConfig,
    CommonConfig,
    LogConfig,
    MetricsConfig,
    ProcessingGraphConfig,
    QueueConfig,
)

CACHE_MAINTENANCE_BACKFILL_ERROR_CODES_TO_RETRY = None


@dataclass(frozen=True)
class BackfillConfig:
    error_codes_to_retry: Optional[List[str]] = CACHE_MAINTENANCE_BACKFILL_ERROR_CODES_TO_RETRY

    @classmethod
    def from_env(cls) -> "BackfillConfig":
        env = Env(expand_vars=True)

        return cls(
            error_codes_to_retry=env.list(name="CACHE_MAINTENANCE_BACKFILL_ERROR_CODES_TO_RETRY", default=""),
        )


DUCKDB_INDEX_CACHE_DIRECTORY = None
DUCKDB_INDEX_SUBDIRECTORY = "downloads"
DUCKDB_INDEX_EXPIRED_TIME_INTERVAL_SECONDS = 10 * 60  # 10 minutes
DUCKDB_INDEX_FILE_EXTENSION = ".duckdb"


@dataclass(frozen=True)
class DuckDbConfig:
    cache_directory: Optional[str] = DUCKDB_INDEX_CACHE_DIRECTORY
    subdirectory: str = DUCKDB_INDEX_SUBDIRECTORY
    expired_time_interval_seconds: int = DUCKDB_INDEX_EXPIRED_TIME_INTERVAL_SECONDS
    file_extension: str = DUCKDB_INDEX_FILE_EXTENSION

    @classmethod
    def from_env(cls) -> "DuckDbConfig":
        env = Env(expand_vars=True)
        with env.prefixed("DUCKDB_INDEX_"):
            return cls(
                cache_directory=env.str(name="CACHE_DIRECTORY", default=DUCKDB_INDEX_CACHE_DIRECTORY),
                subdirectory=env.str(name="SUBDIRECTORY", default=DUCKDB_INDEX_SUBDIRECTORY),
                expired_time_interval_seconds=env.int(
                    name="EXPIRED_TIME_INTERVAL_SECONDS", default=DUCKDB_INDEX_EXPIRED_TIME_INTERVAL_SECONDS
                ),
                file_extension=env.str(name="FILE_EXTENSION", default=DUCKDB_INDEX_FILE_EXTENSION),
            )


CACHE_MAINTENANCE_ACTION = None


@dataclass(frozen=True)
class JobConfig:
    log: LogConfig = field(default_factory=LogConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    graph: ProcessingGraphConfig = field(default_factory=ProcessingGraphConfig)
    backfill: BackfillConfig = field(default_factory=BackfillConfig)
    duckdb: DuckDbConfig = field(default_factory=DuckDbConfig)
    action: Optional[str] = CACHE_MAINTENANCE_ACTION

    @classmethod
    def from_env(cls) -> "JobConfig":
        env = Env(expand_vars=True)

        return cls(
            log=LogConfig.from_env(),
            cache=CacheConfig.from_env(),
            queue=QueueConfig.from_env(),
            metrics=MetricsConfig.from_env(),
            common=CommonConfig.from_env(),
            graph=ProcessingGraphConfig.from_env(),
            backfill=BackfillConfig.from_env(),
            duckdb=DuckDbConfig.from_env(),
            action=env.str(name="CACHE_MAINTENANCE_ACTION", default=CACHE_MAINTENANCE_ACTION),
        )
