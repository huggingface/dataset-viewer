# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from dataclasses import dataclass, field

from libapi.config import ApiConfig
from libcommon.config import (
    CacheConfig,
    CachedAssetsConfig,
    CommonConfig,
    LogConfig,
    ParquetMetadataConfig,
    ProcessingGraphConfig,
    QueueConfig,
    RowsIndexConfig,
)


@dataclass(frozen=True)
class AppConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    cached_assets: CachedAssetsConfig = field(default_factory=CachedAssetsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    log: LogConfig = field(default_factory=LogConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    rows_index: RowsIndexConfig = field(default_factory=RowsIndexConfig)
    processing_graph: ProcessingGraphConfig = field(default_factory=ProcessingGraphConfig)
    parquet_metadata: ParquetMetadataConfig = field(default_factory=ParquetMetadataConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        common_config = CommonConfig.from_env()
        return cls(
            common=common_config,
            cached_assets=CachedAssetsConfig.from_env(),
            cache=CacheConfig.from_env(),
            log=LogConfig.from_env(),
            processing_graph=ProcessingGraphConfig.from_env(),
            queue=QueueConfig.from_env(),
            api=ApiConfig.from_env(hf_endpoint=common_config.hf_endpoint),
            parquet_metadata=ParquetMetadataConfig.from_env(),
            rows_index=RowsIndexConfig.from_env(),
        )
