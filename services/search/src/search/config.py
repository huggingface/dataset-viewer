# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from dataclasses import dataclass, field
from typing import Optional

from environs import Env
from libapi.config import ApiConfig
from libcommon.config import (
    AssetsConfig,
    CacheConfig,
    CachedAssetsConfig,
    CloudFrontConfig,
    CommonConfig,
    LogConfig,
    ParquetMetadataConfig,
    QueueConfig,
    S3Config,
)

DUCKDB_INDEX_CACHE_DIRECTORY = None
DUCKDB_INDEX_CACHE_CLEAN_CACHE_PROBA = 0.05
DUCKDB_INDEX_CACHE_EXPIRED_TIME_INTERVAL_SECONDS = 3_600  # 1 hour

DUCKDB_INDEX_EXTENSIONS_DIRECTORY: Optional[str] = None


@dataclass(frozen=True)
class DuckDbIndexConfig:
    cache_directory: Optional[str] = DUCKDB_INDEX_CACHE_DIRECTORY
    clean_cache_proba: float = DUCKDB_INDEX_CACHE_CLEAN_CACHE_PROBA
    expired_time_interval_seconds: int = DUCKDB_INDEX_CACHE_EXPIRED_TIME_INTERVAL_SECONDS
    extensions_directory: Optional[str] = DUCKDB_INDEX_EXTENSIONS_DIRECTORY

    @classmethod
    def from_env(cls) -> "DuckDbIndexConfig":
        env = Env(expand_vars=True)
        with env.prefixed("DUCKDB_INDEX_"):
            return cls(
                cache_directory=env.str(name="CACHE_DIRECTORY", default=DUCKDB_INDEX_CACHE_DIRECTORY),
                clean_cache_proba=env.float(
                    name="CACHE_CLEAN_CACHE_PROBA", default=DUCKDB_INDEX_CACHE_CLEAN_CACHE_PROBA
                ),
                expired_time_interval_seconds=env.int(
                    name="CACHE_EXPIRED_TIME_INTERVAL_SECONDS",
                    default=DUCKDB_INDEX_CACHE_EXPIRED_TIME_INTERVAL_SECONDS,
                ),
                extensions_directory=env.str(name="EXTENSIONS_DIRECTORY", default=DUCKDB_INDEX_EXTENSIONS_DIRECTORY),
            )


@dataclass(frozen=True)
class AppConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    cached_assets: CachedAssetsConfig = field(default_factory=CachedAssetsConfig)
    cloudfront: CloudFrontConfig = field(default_factory=CloudFrontConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    log: LogConfig = field(default_factory=LogConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    duckdb_index: DuckDbIndexConfig = field(default_factory=DuckDbIndexConfig)
    s3: S3Config = field(default_factory=S3Config)
    parquet_metadata: ParquetMetadataConfig = field(default_factory=ParquetMetadataConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        common_config = CommonConfig.from_env()
        return cls(
            common=common_config,
            assets=AssetsConfig.from_env(),
            cache=CacheConfig.from_env(),
            cached_assets=CachedAssetsConfig.from_env(),
            cloudfront=CloudFrontConfig.from_env(),
            log=LogConfig.from_env(),
            queue=QueueConfig.from_env(),
            api=ApiConfig.from_env(hf_endpoint=common_config.hf_endpoint),
            duckdb_index=DuckDbIndexConfig.from_env(),
            s3=S3Config.from_env(),
            parquet_metadata=ParquetMetadataConfig.from_env(),
        )
