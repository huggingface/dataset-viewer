# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field

from libcommon.config import CacheConfig, LogConfig, QueueConfig


@dataclass(frozen=True)
class JobConfig:
    cache: CacheConfig = field(default_factory=CacheConfig)
    log: LogConfig = field(default_factory=LogConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)

    @classmethod
    def from_env(cls) -> "JobConfig":
        return cls(
            cache=CacheConfig.from_env(),
            log=LogConfig.from_env(),
            queue=QueueConfig.from_env(),
        )
