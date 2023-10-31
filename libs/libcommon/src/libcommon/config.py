# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.


import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from environs import Env

from libcommon.constants import (
    MIN_BYTES_FOR_BONUS_DIFFICULTY,
    PROCESSING_STEP_CONFIG_INFO_VERSION,
    PROCESSING_STEP_CONFIG_IS_VALID_VERSION,
    PROCESSING_STEP_CONFIG_OPT_IN_OUT_URLS_COUNT_VERSION,
    PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_VERSION,
    PROCESSING_STEP_CONFIG_PARQUET_METADATA_VERSION,
    PROCESSING_STEP_CONFIG_PARQUET_VERSION,
    PROCESSING_STEP_CONFIG_SIZE_VERSION,
    PROCESSING_STEP_CONFIG_SPLIT_NAMES_FROM_INFO_VERSION,
    PROCESSING_STEP_CONFIG_SPLIT_NAMES_FROM_STREAMING_VERSION,
    PROCESSING_STEP_DATASET_CONFIG_NAMES_VERSION,
    PROCESSING_STEP_DATASET_HUB_CACHE_VERSION,
    PROCESSING_STEP_DATASET_INFO_VERSION,
    PROCESSING_STEP_DATASET_IS_VALID_VERSION,
    PROCESSING_STEP_DATASET_OPT_IN_OUT_URLS_COUNT_VERSION,
    PROCESSING_STEP_DATASET_PARQUET_VERSION,
    PROCESSING_STEP_DATASET_SIZE_VERSION,
    PROCESSING_STEP_DATASET_SPLIT_NAMES_VERSION,
    PROCESSING_STEP_SPLIT_DESCRIPTIVE_STATISTICS_VERSION,
    PROCESSING_STEP_SPLIT_DUCKDB_INDEX_VERSION,
    PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_PARQUET_VERSION,
    PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_STREAMING_VERSION,
    PROCESSING_STEP_SPLIT_IMAGE_URL_COLUMNS_VERSION,
    PROCESSING_STEP_SPLIT_IS_VALID_VERSION,
    PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_COUNT_VERSION,
    PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION,
)

if TYPE_CHECKING:
    from libcommon.processing_graph import ProcessingGraphSpecification

ASSETS_BASE_URL = "http://localhost/assets"
ASSETS_FOLDER_NAME = "assets"
ASSETS_STORAGE_ROOT = "/storage"
ASSETS_STORAGE_PROTOCOL = "file"


@dataclass(frozen=True)
class AssetsConfig:
    base_url: str = ASSETS_BASE_URL
    folder_name: str = ASSETS_FOLDER_NAME
    storage_protocol: str = ASSETS_STORAGE_PROTOCOL
    storage_root: str = ASSETS_STORAGE_ROOT

    @classmethod
    def from_env(cls) -> "AssetsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("ASSETS_"):
            return cls(
                base_url=env.str(name="BASE_URL", default=ASSETS_BASE_URL),
                folder_name=env.str(name="FOLDER_NAME", default=ASSETS_FOLDER_NAME),
                storage_protocol=env.str(name="STORAGE_PROTOCOL", default=ASSETS_STORAGE_PROTOCOL),
                storage_root=env.str(name="STORAGE_ROOT", default=ASSETS_STORAGE_ROOT),
            )


S3_ACCESS_KEY_ID = None
S3_SECRET_ACCESS_KEY = None
S3_REGION_NAME = "us-east-1"


@dataclass(frozen=True)
class S3Config:
    access_key_id: Optional[str] = S3_ACCESS_KEY_ID
    secret_access_key: Optional[str] = S3_SECRET_ACCESS_KEY
    region_name: str = S3_REGION_NAME

    @classmethod
    def from_env(cls) -> "S3Config":
        env = Env(expand_vars=True)
        with env.prefixed("S3_"):
            return cls(
                access_key_id=env.str(name="ACCESS_KEY_ID", default=S3_ACCESS_KEY_ID),
                secret_access_key=env.str(name="SECRET_ACCESS_KEY", default=S3_SECRET_ACCESS_KEY),
                region_name=env.str(name="REGION_NAME", default=S3_REGION_NAME),
            )


CACHED_ASSETS_BASE_URL = "http://localhost/cached-assets"
CACHED_ASSETS_FOLDER_NAME = "cached-assets"
CACHED_ASSETS_STORAGE_ROOT = "/storage"
CACHED_ASSETS_STORAGE_PROTOCOL = "file"


@dataclass(frozen=True)
class CachedAssetsConfig:
    base_url: str = CACHED_ASSETS_BASE_URL
    folder_name: str = CACHED_ASSETS_FOLDER_NAME
    storage_protocol: str = CACHED_ASSETS_STORAGE_PROTOCOL
    storage_root: str = CACHED_ASSETS_STORAGE_ROOT

    @classmethod
    def from_env(cls) -> "CachedAssetsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("CACHED_ASSETS_"):
            return cls(
                base_url=env.str(name="BASE_URL", default=CACHED_ASSETS_BASE_URL),
                folder_name=env.str(name="FOLDER_NAME", default=CACHED_ASSETS_FOLDER_NAME),
                storage_protocol=env.str(name="STORAGE_PROTOCOL", default=CACHED_ASSETS_STORAGE_PROTOCOL),
                storage_root=env.str(name="STORAGE_ROOT", default=CACHED_ASSETS_STORAGE_ROOT),
            )


PARQUET_METADATA_STORAGE_DIRECTORY = None


@dataclass(frozen=True)
class ParquetMetadataConfig:
    storage_directory: Optional[str] = PARQUET_METADATA_STORAGE_DIRECTORY

    @classmethod
    def from_env(cls) -> "ParquetMetadataConfig":
        env = Env(expand_vars=True)
        with env.prefixed("PARQUET_METADATA_"):
            return cls(
                storage_directory=env.str(name="STORAGE_DIRECTORY", default=PARQUET_METADATA_STORAGE_DIRECTORY),
            )


ROWS_INDEX_MAX_ARROW_DATA_IN_MEMORY = 300_000_000


@dataclass(frozen=True)
class RowsIndexConfig:
    max_arrow_data_in_memory: int = ROWS_INDEX_MAX_ARROW_DATA_IN_MEMORY

    @classmethod
    def from_env(cls) -> "RowsIndexConfig":
        env = Env(expand_vars=True)
        with env.prefixed("ROWS_INDEX_"):
            return cls(
                max_arrow_data_in_memory=env.int(
                    name="MAX_ARROW_DATA_IN_MEMORY", default=ROWS_INDEX_MAX_ARROW_DATA_IN_MEMORY
                ),
            )


COMMON_BLOCKED_DATASETS: list[str] = []
COMMON_DATASET_SCRIPTS_ALLOW_LIST: list[str] = []
COMMON_HF_ENDPOINT = "https://huggingface.co"
COMMON_HF_TOKEN = None


@dataclass(frozen=True)
class CommonConfig:
    blocked_datasets: list[str] = field(default_factory=COMMON_BLOCKED_DATASETS.copy)
    dataset_scripts_allow_list: list[str] = field(default_factory=COMMON_DATASET_SCRIPTS_ALLOW_LIST.copy)
    hf_endpoint: str = COMMON_HF_ENDPOINT
    hf_token: Optional[str] = COMMON_HF_TOKEN

    @classmethod
    def from_env(cls) -> "CommonConfig":
        env = Env(expand_vars=True)
        with env.prefixed("COMMON_"):
            return cls(
                blocked_datasets=env.list(name="BLOCKED_DATASETS", default=COMMON_BLOCKED_DATASETS.copy()),
                dataset_scripts_allow_list=env.list(
                    name="DATASET_SCRIPTS_ALLOW_LIST", default=COMMON_DATASET_SCRIPTS_ALLOW_LIST.copy()
                ),
                hf_endpoint=env.str(name="HF_ENDPOINT", default=COMMON_HF_ENDPOINT),
                hf_token=env.str(name="HF_TOKEN", default=COMMON_HF_TOKEN),  # nosec
            )


LOG_LEVEL = logging.INFO


@dataclass(frozen=True)
class LogConfig:
    level: int = LOG_LEVEL

    @classmethod
    def from_env(cls) -> "LogConfig":
        env = Env(expand_vars=True)
        with env.prefixed("LOG_"):
            return cls(
                level=env.log_level(name="LEVEL", default=LOG_LEVEL),
            )


CACHE_MAX_DAYS = 90  # 3 months
CACHE_MONGO_DATABASE = "datasets_server_cache"
CACHE_MONGO_URL = "mongodb://localhost:27017"


@dataclass(frozen=True)
class CacheConfig:
    max_days: int = CACHE_MAX_DAYS
    mongo_database: str = CACHE_MONGO_DATABASE
    mongo_url: str = CACHE_MONGO_URL

    @classmethod
    def from_env(cls) -> "CacheConfig":
        env = Env(expand_vars=True)
        with env.prefixed("CACHE_"):
            return cls(
                max_days=env.int(name="MAX_DAYS", default=CACHE_MAX_DAYS),
                mongo_database=env.str(name="MONGO_DATABASE", default=CACHE_MONGO_DATABASE),
                mongo_url=env.str(name="MONGO_URL", default=CACHE_MONGO_URL),
            )


QUEUE_MONGO_DATABASE = "datasets_server_queue"
QUEUE_MONGO_URL = "mongodb://localhost:27017"


@dataclass(frozen=True)
class QueueConfig:
    mongo_database: str = QUEUE_MONGO_DATABASE
    mongo_url: str = QUEUE_MONGO_URL

    @classmethod
    def from_env(cls) -> "QueueConfig":
        env = Env(expand_vars=True)
        with env.prefixed("QUEUE_"):
            return cls(
                mongo_database=env.str(name="MONGO_DATABASE", default=QUEUE_MONGO_DATABASE),
                mongo_url=env.str(name="MONGO_URL", default=QUEUE_MONGO_URL),
            )


@dataclass(frozen=True)
class ProcessingGraphConfig:
    specification: "ProcessingGraphSpecification" = field(
        default_factory=lambda: {
            "dataset-config-names": {
                "input_type": "dataset",
                "provides_dataset_config_names": True,
                "job_runner_version": PROCESSING_STEP_DATASET_CONFIG_NAMES_VERSION,
                "difficulty": 50,
            },
            "config-split-names-from-streaming": {
                "input_type": "config",
                "triggered_by": "dataset-config-names",
                "provides_config_split_names": True,
                "job_runner_version": PROCESSING_STEP_CONFIG_SPLIT_NAMES_FROM_STREAMING_VERSION,
                "difficulty": 60,
            },
            "split-first-rows-from-streaming": {
                "input_type": "split",
                "triggered_by": ["config-split-names-from-streaming", "config-split-names-from-info"],
                "enables_preview": True,
                "job_runner_version": PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_STREAMING_VERSION,
                "difficulty": 70,
            },
            "config-parquet-and-info": {
                "input_type": "config",
                "triggered_by": "dataset-config-names",
                "job_runner_version": PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_VERSION,
                "difficulty": 70,
            },
            "config-parquet": {
                "input_type": "config",
                "triggered_by": "config-parquet-and-info",
                "job_runner_version": PROCESSING_STEP_CONFIG_PARQUET_VERSION,
                "provides_config_parquet": True,
                "difficulty": 20,
            },
            "config-parquet-metadata": {
                "input_type": "config",
                "triggered_by": "config-parquet",
                "job_runner_version": PROCESSING_STEP_CONFIG_PARQUET_METADATA_VERSION,
                "provides_config_parquet_metadata": True,
                "difficulty": 50,
            },
            "split-first-rows-from-parquet": {
                "input_type": "split",
                "triggered_by": "config-parquet-metadata",
                "enables_preview": True,
                "job_runner_version": PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_PARQUET_VERSION,
                "difficulty": 40,
            },
            "dataset-parquet": {
                "input_type": "dataset",
                "triggered_by": ["config-parquet", "dataset-config-names"],
                "job_runner_version": PROCESSING_STEP_DATASET_PARQUET_VERSION,
                "difficulty": 20,
            },
            "config-info": {
                "input_type": "config",
                "triggered_by": "config-parquet-and-info",
                "job_runner_version": PROCESSING_STEP_CONFIG_INFO_VERSION,
                "difficulty": 20,
                "provides_config_info": True,
            },
            "dataset-info": {
                "input_type": "dataset",
                "triggered_by": ["config-info", "dataset-config-names"],
                "job_runner_version": PROCESSING_STEP_DATASET_INFO_VERSION,
                "difficulty": 20,
            },
            "config-split-names-from-info": {
                "input_type": "config",
                "triggered_by": "config-info",
                "provides_config_split_names": True,
                "job_runner_version": PROCESSING_STEP_CONFIG_SPLIT_NAMES_FROM_INFO_VERSION,
                "difficulty": 20,
            },
            "config-size": {
                "input_type": "config",
                "triggered_by": "config-parquet-and-info",
                "enables_viewer": True,
                "job_runner_version": PROCESSING_STEP_CONFIG_SIZE_VERSION,
                "difficulty": 20,
            },
            "dataset-size": {
                "input_type": "dataset",
                "triggered_by": ["config-size", "dataset-config-names"],
                "job_runner_version": PROCESSING_STEP_DATASET_SIZE_VERSION,
                "difficulty": 20,
            },
            "dataset-split-names": {
                "input_type": "dataset",
                "triggered_by": [
                    "config-split-names-from-info",
                    "config-split-names-from-streaming",
                    "dataset-config-names",
                ],
                "job_runner_version": PROCESSING_STEP_DATASET_SPLIT_NAMES_VERSION,
                "difficulty": 20,
            },
            "split-descriptive-statistics": {
                "input_type": "split",
                "triggered_by": [
                    "config-split-names-from-info",
                    "config-split-names-from-streaming",
                ],
                "job_runner_version": PROCESSING_STEP_SPLIT_DESCRIPTIVE_STATISTICS_VERSION,
                "difficulty": 70,
                "bonus_difficulty_if_dataset_is_big": 20,
            },
            "split-is-valid": {
                "input_type": "split",
                # special case: triggered by all the steps that have enables_preview/enables_viewer/enables_search
                "triggered_by": [
                    "config-size",
                    "split-first-rows-from-parquet",
                    "split-first-rows-from-streaming",
                    "split-duckdb-index",
                ],
                "job_runner_version": PROCESSING_STEP_SPLIT_IS_VALID_VERSION,
                "difficulty": 20,
            },
            "config-is-valid": {
                "input_type": "config",
                "triggered_by": [
                    "config-split-names-from-streaming",
                    "config-split-names-from-info",
                    "split-is-valid",
                ],
                "job_runner_version": PROCESSING_STEP_CONFIG_IS_VALID_VERSION,
                "difficulty": 20,
            },
            "dataset-is-valid": {
                "input_type": "dataset",
                "triggered_by": [
                    "dataset-config-names",
                    "config-is-valid",
                ],
                "job_runner_version": PROCESSING_STEP_DATASET_IS_VALID_VERSION,
                "difficulty": 20,
            },
            "split-image-url-columns": {
                "input_type": "split",
                "triggered_by": ["split-first-rows-from-streaming", "split-first-rows-from-parquet"],
                "job_runner_version": PROCESSING_STEP_SPLIT_IMAGE_URL_COLUMNS_VERSION,
                "difficulty": 40,
            },
            "split-opt-in-out-urls-scan": {
                "input_type": "split",
                "triggered_by": ["split-image-url-columns"],
                "job_runner_version": PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION,
                "difficulty": 70,
            },
            "split-opt-in-out-urls-count": {
                "input_type": "split",
                "triggered_by": ["split-opt-in-out-urls-scan"],
                "job_runner_version": PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_COUNT_VERSION,
                "difficulty": 20,
            },
            "config-opt-in-out-urls-count": {
                "input_type": "config",
                "triggered_by": [
                    "config-split-names-from-streaming",
                    "config-split-names-from-info",
                    "split-opt-in-out-urls-count",
                ],
                "job_runner_version": PROCESSING_STEP_CONFIG_OPT_IN_OUT_URLS_COUNT_VERSION,
                "difficulty": 20,
            },
            "dataset-opt-in-out-urls-count": {
                "input_type": "dataset",
                "triggered_by": ["dataset-config-names", "config-opt-in-out-urls-count"],
                "job_runner_version": PROCESSING_STEP_DATASET_OPT_IN_OUT_URLS_COUNT_VERSION,
                "difficulty": 20,
            },
            "split-duckdb-index": {
                "input_type": "split",
                "triggered_by": [
                    "config-split-names-from-info",
                    "config-split-names-from-streaming",
                    "config-parquet-and-info",
                ],
                "enables_search": True,
                "job_runner_version": PROCESSING_STEP_SPLIT_DUCKDB_INDEX_VERSION,
                "difficulty": 70,
                "bonus_difficulty_if_dataset_is_big": 20,
            },
            "dataset-hub-cache": {
                "input_type": "dataset",
                "triggered_by": ["dataset-is-valid", "dataset-size"],
                "job_runner_version": PROCESSING_STEP_DATASET_HUB_CACHE_VERSION,
                "difficulty": 20,
            },
        }
    )
    min_bytes_for_bonus_difficulty: int = MIN_BYTES_FOR_BONUS_DIFFICULTY

    @classmethod
    def from_env(cls) -> "ProcessingGraphConfig":
        # TODO: allow passing the graph via env vars
        return cls()
