# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field
from typing import Optional

from environs import Env
from libcommon.config import CommonConfig, LogConfig, ProcessingGraphConfig, QueueConfig

CACHE_MAINTENANCE_ACTION = None


@dataclass(frozen=True)
class JobConfig:
    log: LogConfig = field(default_factory=LogConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    graph: ProcessingGraphConfig = field(default_factory=ProcessingGraphConfig)
    action: Optional[str] = CACHE_MAINTENANCE_ACTION

    @classmethod
    def from_env(cls) -> "JobConfig":
        env = Env(expand_vars=True)

        return cls(
            log=LogConfig.from_env(),
            queue=QueueConfig.from_env(),
            common=CommonConfig.from_env(),
            graph=ProcessingGraphConfig.from_env(),
            action=env.str(name="CACHE_MAINTENANCE_ACTION", default=CACHE_MAINTENANCE_ACTION),
        )
