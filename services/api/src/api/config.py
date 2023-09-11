# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field
from typing import List, Mapping

from environs import Env
from libapi.config import ApiConfig
from libcommon.config import (
    CacheConfig,
    CachedAssetsConfig,
    CommonConfig,
    LogConfig,
    ParquetMetadataConfig,
    ProcessingGraphConfig,
    QueueConfig,
)
from libcommon.processing_graph import InputType

HUB_CACHE_BASE_URL = "https://datasets-server.huggingface.co"
HUB_CACHE_CACHE_KIND = "dataset-hub-cache"
HUB_CACHE_NUM_RESULTS_PER_PAGE = 1_000


@dataclass(frozen=True)
class HubCacheConfig:
    base_url: str = HUB_CACHE_BASE_URL
    cache_kind: str = HUB_CACHE_CACHE_KIND
    num_results_per_page: int = HUB_CACHE_NUM_RESULTS_PER_PAGE

    @classmethod
    def from_env(cls) -> "HubCacheConfig":
        env = Env(expand_vars=True)
        with env.prefixed("HUB_CACHE_"):
            return cls(
                base_url=env.str(name="BASE_URL", default=HUB_CACHE_BASE_URL),
                cache_kind=HUB_CACHE_CACHE_KIND,  # don't allow changing the cache kind
                num_results_per_page=env.int(name="NUM_RESULTS_PER_PAGE", default=HUB_CACHE_NUM_RESULTS_PER_PAGE),
            )


@dataclass(frozen=True)
class AppConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    log: LogConfig = field(default_factory=LogConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    processing_graph: ProcessingGraphConfig = field(default_factory=ProcessingGraphConfig)
    hub_cache: HubCacheConfig = field(default_factory=HubCacheConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        common_config = CommonConfig.from_env()
        return cls(
            common=common_config,
            cache=CacheConfig.from_env(),
            log=LogConfig.from_env(),
            processing_graph=ProcessingGraphConfig.from_env(),
            queue=QueueConfig.from_env(),
            api=ApiConfig.from_env(hf_endpoint=common_config.hf_endpoint),
            hub_cache=HubCacheConfig.from_env(),
        )


@dataclass(frozen=True)
class FilterAppConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    cached_assets: CachedAssetsConfig = field(default_factory=CachedAssetsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    log: LogConfig = field(default_factory=LogConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    processing_graph: ProcessingGraphConfig = field(default_factory=ProcessingGraphConfig)
    parquet_metadata: ParquetMetadataConfig = field(default_factory=ParquetMetadataConfig)

    @classmethod
    def from_env(cls) -> "FilterAppConfig":
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
        )


ProcessingStepNamesByInputType = Mapping[InputType, List[str]]

ProcessingStepNamesByInputTypeAndEndpoint = Mapping[str, ProcessingStepNamesByInputType]


@dataclass(frozen=True)
class EndpointConfig:
    """Contains the endpoint config specification to relate with step names.
    The list of processing steps corresponds to the priority in which the response
    has to be reached. The cache from the first step in the list will be used first
    then, if it's an error or missing, the second one, etc.
    The related steps depend on the query parameters passed in the request
    (dataset, config, split)
    """

    processing_step_names_by_input_type_and_endpoint: ProcessingStepNamesByInputTypeAndEndpoint = field(
        default_factory=lambda: {
            "/splits": {
                "dataset": [
                    "dataset-split-names",
                ],
                "config": ["config-split-names-from-streaming", "config-split-names-from-info"],
            },
            "/first-rows": {"split": ["split-first-rows-from-streaming", "split-first-rows-from-parquet"]},
            "/parquet": {
                "dataset": ["dataset-parquet"],
                "config": ["config-parquet"],
            },
            "/info": {"dataset": ["dataset-info"], "config": ["config-info"]},
            "/size": {
                "dataset": ["dataset-size"],
                "config": ["config-size"],
            },
            "/opt-in-out-urls": {
                "dataset": ["dataset-opt-in-out-urls-count"],
                "config": ["config-opt-in-out-urls-count"],
                "split": ["split-opt-in-out-urls-count"],
            },
            "/is-valid": {
                "dataset": ["dataset-is-valid"],
                "config": ["config-is-valid"],
                "split": ["split-is-valid"],
            },
            "/statistics": {"split": ["split-descriptive-statistics"]},
        }
    )

    @classmethod
    def from_env(cls) -> "EndpointConfig":
        # TODO: allow passing the mapping between endpoint and processing steps via env vars
        return cls()
