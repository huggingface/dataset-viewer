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
            action=env.str(name="CACHE_MAINTENANCE_ACTION", default=CACHE_MAINTENANCE_ACTION),
        )
