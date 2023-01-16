# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.


from dataclasses import dataclass, field

from environs import Env
from libcommon.config import (
    CacheConfig,
    CommonConfig,
    ProcessingGraphConfig,
    QueueConfig,
    WorkerLoopConfig,
)

PARQUET_BASED_ENDPOINT = "/size"


@dataclass
class ParquetBasedConfig:
    endpoint: str = PARQUET_BASED_ENDPOINT

    @staticmethod
    def from_env() -> "ParquetBasedConfig":
        env = Env(expand_vars=True)
        with env.prefixed("PARQUET_BASED_"):
            return ParquetBasedConfig(
                endpoint=env.str(name="ENDPOINT", default=PARQUET_BASED_ENDPOINT),
            )


@dataclass
class AppConfig:
    cache: CacheConfig = field(default_factory=CacheConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    parquet_based: ParquetBasedConfig = field(default_factory=ParquetBasedConfig)
    processing_graph: ProcessingGraphConfig = field(default_factory=ProcessingGraphConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    worker_loop: WorkerLoopConfig = field(default_factory=WorkerLoopConfig)

    @staticmethod
    def from_env() -> "AppConfig":
        return AppConfig(
            # First process the common configuration to setup the logging
            common=CommonConfig.from_env(),
            cache=CacheConfig.from_env(),
            parquet_based=ParquetBasedConfig.from_env(),
            processing_graph=ProcessingGraphConfig.from_env(),
            queue=QueueConfig.from_env(),
            worker_loop=WorkerLoopConfig.from_env(),
        )
