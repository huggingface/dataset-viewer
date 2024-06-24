# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field

from libapi.config import ApiConfig
from libcommon.config import (
    AssetsConfig,
    CacheConfig,
    CachedAssetsConfig,
    CloudFrontConfig,
    CommonConfig,
    LogConfig,
    QueueConfig,
    S3Config,
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
    s3: S3Config = field(default_factory=S3Config)

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
            s3=S3Config.from_env(),
        )
