# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from dataclasses import dataclass, field
from typing import Optional

from environs import Env
from libapi.config import ApiConfig, CachedAssetsS3Config
from libcommon.config import (
    CacheConfig,
    CachedAssetsConfig,
    CommonConfig,
    LogConfig,
    ProcessingGraphConfig,
    QueueConfig,
)

DUCKDB_INDEX_CACHE_DIRECTORY = None
DUCKDB_INDEX_TARGET_REVISION = "refs/convert/parquet"


@dataclass(frozen=True)
class DuckDbIndexConfig:
    cache_directory: Optional[str] = DUCKDB_INDEX_CACHE_DIRECTORY
    target_revision: str = DUCKDB_INDEX_TARGET_REVISION

    @classmethod
    def from_env(cls) -> "DuckDbIndexConfig":
        env = Env(expand_vars=True)
        with env.prefixed("DUCKDB_INDEX_"):
            return cls(
                cache_directory=env.str(name="CACHE_DIRECTORY", default=DUCKDB_INDEX_CACHE_DIRECTORY),
                target_revision=env.str(name="TARGET_REVISION", default=DUCKDB_INDEX_TARGET_REVISION),
            )


@dataclass(frozen=True)
class AppConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    cached_assets: CachedAssetsConfig = field(default_factory=CachedAssetsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    cached_assets_s3: CachedAssetsS3Config = field(default_factory=CachedAssetsS3Config)
    common: CommonConfig = field(default_factory=CommonConfig)
    log: LogConfig = field(default_factory=LogConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    processing_graph: ProcessingGraphConfig = field(default_factory=ProcessingGraphConfig)
    duckdb_index: DuckDbIndexConfig = field(default_factory=DuckDbIndexConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        common_config = CommonConfig.from_env()
        return cls(
            common=common_config,
            cached_assets=CachedAssetsConfig.from_env(),
            cached_assets_s3=CachedAssetsS3Config.from_env(),
            cache=CacheConfig.from_env(),
            log=LogConfig.from_env(),
            processing_graph=ProcessingGraphConfig.from_env(),
            queue=QueueConfig.from_env(),
            api=ApiConfig.from_env(hf_endpoint=common_config.hf_endpoint),
            duckdb_index=DuckDbIndexConfig.from_env(),
        )
