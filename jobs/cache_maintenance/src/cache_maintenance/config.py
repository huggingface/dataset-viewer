# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field
from typing import Optional

from environs import Env
from libcommon.config import (
    CacheConfig,
    CommonConfig,
    LogConfig,
    ProcessingGraphConfig,
    QueueConfig,
)

CACHE_MAINTENANCE_BACKFILL_ERROR_CODES_TO_RETRY = None


@dataclass(frozen=True)
class BackfillConfig:
    error_codes_to_retry: Optional[list[str]] = CACHE_MAINTENANCE_BACKFILL_ERROR_CODES_TO_RETRY

    @classmethod
    def from_env(cls) -> "BackfillConfig":
        env = Env(expand_vars=True)

        return cls(
            error_codes_to_retry=env.list(name="CACHE_MAINTENANCE_BACKFILL_ERROR_CODES_TO_RETRY", default=""),
        )


DISCUSSIONS_BOT_ASSOCIATED_USER_NAME = None
DISCUSSIONS_BOT_TOKEN = None
DISCUSSIONS_PARQUET_REVISION = "refs/convert/parquet"


@dataclass(frozen=True)
class DiscussionsConfig:
    bot_associated_user_name: Optional[str] = DISCUSSIONS_BOT_ASSOCIATED_USER_NAME
    bot_token: Optional[str] = DISCUSSIONS_BOT_TOKEN
    parquet_revision: str = DISCUSSIONS_PARQUET_REVISION

    @classmethod
    def from_env(cls) -> "DiscussionsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("DISCUSSIONS_"):
            return cls(
                bot_associated_user_name=env.str(
                    name="BOT_ASSOCIATED_USER_NAME", default=DISCUSSIONS_BOT_ASSOCIATED_USER_NAME
                ),
                bot_token=env.str(name="BOT_TOKEN", default=DISCUSSIONS_BOT_TOKEN),
                parquet_revision=env.str(name="PARQUET_REVISION", default=DISCUSSIONS_PARQUET_REVISION),
            )


DIRECTORY_CLEANING_CACHE_DIRECTORY = None
DIRECTORY_CLEANING_SUBFOLDER_PATTERN = "*/datasets/*"
DIRECTORY_CLEANING_EXPIRED_TIME_INTERVAL_SECONDS = 3 * 60 * 60  # 3 hours


@dataclass(frozen=True)
class DirectoryCleaning:
    cache_directory: Optional[str] = DIRECTORY_CLEANING_CACHE_DIRECTORY
    subfolder_pattern: str = DIRECTORY_CLEANING_SUBFOLDER_PATTERN
    expired_time_interval_seconds: int = DIRECTORY_CLEANING_EXPIRED_TIME_INTERVAL_SECONDS

    @classmethod
    def from_env(cls) -> "DirectoryCleaning":
        env = Env(expand_vars=True)
        with env.prefixed("DIRECTORY_CLEANING_"):
            return cls(
                cache_directory=env.str(name="CACHE_DIRECTORY", default=DIRECTORY_CLEANING_CACHE_DIRECTORY),
                subfolder_pattern=env.str(name="SUBFOLDER_PATTERN", default=DIRECTORY_CLEANING_SUBFOLDER_PATTERN),
                expired_time_interval_seconds=env.int(
                    name="EXPIRED_TIME_INTERVAL_SECONDS", default=DIRECTORY_CLEANING_EXPIRED_TIME_INTERVAL_SECONDS
                ),
            )


CACHE_MAINTENANCE_ACTION = None


@dataclass(frozen=True)
class JobConfig:
    log: LogConfig = field(default_factory=LogConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    graph: ProcessingGraphConfig = field(default_factory=ProcessingGraphConfig)
    backfill: BackfillConfig = field(default_factory=BackfillConfig)
    directory_cleaning: DirectoryCleaning = field(default_factory=DirectoryCleaning)
    discussions: DiscussionsConfig = field(default_factory=DiscussionsConfig)
    action: Optional[str] = CACHE_MAINTENANCE_ACTION

    @classmethod
    def from_env(cls) -> "JobConfig":
        env = Env(expand_vars=True)

        return cls(
            log=LogConfig.from_env(),
            cache=CacheConfig.from_env(),
            queue=QueueConfig.from_env(),
            common=CommonConfig.from_env(),
            graph=ProcessingGraphConfig.from_env(),
            backfill=BackfillConfig.from_env(),
            directory_cleaning=DirectoryCleaning.from_env(),
            discussions=DiscussionsConfig.from_env(),
            action=env.str(name="CACHE_MAINTENANCE_ACTION", default=CACHE_MAINTENANCE_ACTION),
        )
