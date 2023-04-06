# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.


import logging
from dataclasses import dataclass, field
from typing import Optional

from environs import Env

from libcommon.constants import (
    PROCESSING_STEP_CONFIG_INFO_VERSION,
    PROCESSING_STEP_CONFIG_NAMES_VERSION,
    PROCESSING_STEP_CONFIG_PARQUET_VERSION,
    PROCESSING_STEP_CONFIG_SIZE_VERSION,
    PROCESSING_STEP_DATASET_INFO_VERSION,
    PROCESSING_STEP_DATASET_PARQUET_VERSION,
    PROCESSING_STEP_DATASET_SIZE_VERSION,
    PROCESSING_STEP_DATASET_SPLIT_NAMES_FROM_DATASET_INFO_VERSION,
    PROCESSING_STEP_DATASET_SPLIT_NAMES_FROM_STREAMING_VERSION,
    PROCESSING_STEP_PARQUET_AND_DATASET_INFO_VERSION,
    PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_PARQUET_VERSION,
    PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_STREAMING_VERSION,
    PROCESSING_STEP_SPLIT_NAMES_FROM_DATASET_INFO_VERSION,
    PROCESSING_STEP_SPLIT_NAMES_FROM_STREAMING_VERSION,
    PROCESSING_STEP_SPLITS_VERSION,
)
from libcommon.processing_graph import ProcessingGraphSpecification

ASSETS_BASE_URL = "assets"
ASSETS_STORAGE_DIRECTORY = None


@dataclass(frozen=True)
class AssetsConfig:
    base_url: str = ASSETS_BASE_URL
    storage_directory: Optional[str] = ASSETS_STORAGE_DIRECTORY

    @classmethod
    def from_env(cls) -> "AssetsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("ASSETS_"):
            return cls(
                base_url=env.str(name="BASE_URL", default=ASSETS_BASE_URL),
                storage_directory=env.str(name="STORAGE_DIRECTORY", default=ASSETS_STORAGE_DIRECTORY),
            )


CACHED_ASSETS_BASE_URL = "cached-assets"
CACHED_ASSETS_STORAGE_DIRECTORY = None
CACHED_ASSETS_CLEAN_CACHE_PROBA = 0.05
CACHED_ASSETS_KEEP_ROWS_BELOW_INDEX = 100
CACHED_ASSETS_KEEP_N_MOST_RECENT_ROWS = 200
CACHED_ASSETS_MAX_CLEAN_SAMPLE_SIZE = 10_000


@dataclass(frozen=True)
class CachedAssetsConfig:
    base_url: str = ASSETS_BASE_URL
    storage_directory: Optional[str] = ASSETS_STORAGE_DIRECTORY
    clean_cache_proba: float = CACHED_ASSETS_CLEAN_CACHE_PROBA
    keep_rows_below_index: int = CACHED_ASSETS_KEEP_ROWS_BELOW_INDEX
    keep_n_most_recent_rows: int = CACHED_ASSETS_KEEP_N_MOST_RECENT_ROWS
    max_clean_sample_size: int = CACHED_ASSETS_MAX_CLEAN_SAMPLE_SIZE

    @classmethod
    def from_env(cls) -> "CachedAssetsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("CACHED_ASSETS_"):
            return cls(
                base_url=env.str(name="BASE_URL", default=CACHED_ASSETS_BASE_URL),
                storage_directory=env.str(name="STORAGE_DIRECTORY", default=CACHED_ASSETS_STORAGE_DIRECTORY),
                clean_cache_proba=env.float(name="CLEAN_CACHE_PROBA", default=CACHED_ASSETS_CLEAN_CACHE_PROBA),
                keep_rows_below_index=env.float(
                    name="KEEP_ROWS_BELOW_INDEX", default=CACHED_ASSETS_KEEP_ROWS_BELOW_INDEX
                ),
                keep_n_most_recent_rows=env.float(
                    name="KEEP_N_MOST_RECENT_ROWS", default=CACHED_ASSETS_KEEP_N_MOST_RECENT_ROWS
                ),
                max_clean_sample_size=env.float(
                    name="MAX_CLEAN_SAMPLE_SIZE", default=CACHED_ASSETS_MAX_CLEAN_SAMPLE_SIZE
                ),
            )


COMMON_HF_ENDPOINT = "https://huggingface.co"
COMMON_HF_TOKEN = None


@dataclass(frozen=True)
class CommonConfig:
    hf_endpoint: str = COMMON_HF_ENDPOINT
    hf_token: Optional[str] = COMMON_HF_TOKEN

    @classmethod
    def from_env(cls) -> "CommonConfig":
        env = Env(expand_vars=True)
        with env.prefixed("COMMON_"):
            return cls(
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


CACHE_MONGO_DATABASE = "datasets_server_cache"
CACHE_MONGO_URL = "mongodb://localhost:27017"


@dataclass(frozen=True)
class CacheConfig:
    mongo_database: str = CACHE_MONGO_DATABASE
    mongo_url: str = CACHE_MONGO_URL

    @classmethod
    def from_env(cls) -> "CacheConfig":
        env = Env(expand_vars=True)
        with env.prefixed("CACHE_"):
            return cls(
                mongo_database=env.str(name="MONGO_DATABASE", default=CACHE_MONGO_DATABASE),
                mongo_url=env.str(name="MONGO_URL", default=CACHE_MONGO_URL),
            )


QUEUE_MAX_JOBS_PER_NAMESPACE = 1
QUEUE_MONGO_DATABASE = "datasets_server_queue"
QUEUE_MONGO_URL = "mongodb://localhost:27017"


@dataclass(frozen=True)
class QueueConfig:
    max_jobs_per_namespace: int = QUEUE_MAX_JOBS_PER_NAMESPACE
    mongo_database: str = QUEUE_MONGO_DATABASE
    mongo_url: str = QUEUE_MONGO_URL

    @classmethod
    def from_env(cls) -> "QueueConfig":
        env = Env(expand_vars=True)
        with env.prefixed("QUEUE_"):
            return cls(
                max_jobs_per_namespace=env.int(name="MAX_JOBS_PER_NAMESPACE", default=QUEUE_MAX_JOBS_PER_NAMESPACE),
                mongo_database=env.str(name="MONGO_DATABASE", default=QUEUE_MONGO_DATABASE),
                mongo_url=env.str(name="MONGO_URL", default=QUEUE_MONGO_URL),
            )


@dataclass(frozen=True)
class ProcessingGraphConfig:
    specification: ProcessingGraphSpecification = field(
        default_factory=lambda: {
            "/config-names": {"input_type": "dataset", "job_runner_version": PROCESSING_STEP_CONFIG_NAMES_VERSION},
            "/split-names-from-streaming": {
                "input_type": "config",
                "requires": "/config-names",
                "job_runner_version": PROCESSING_STEP_SPLIT_NAMES_FROM_STREAMING_VERSION,
            },
            "/splits": {
                "input_type": "dataset",
                "required_by_dataset_viewer": True,
                "job_runner_version": PROCESSING_STEP_SPLITS_VERSION,
            },  # to be deprecated
            "split-first-rows-from-streaming": {
                "input_type": "split",
                "requires": ["/split-names-from-streaming", "/split-names-from-dataset-info"],
                "required_by_dataset_viewer": True,
                "job_runner_version": PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_STREAMING_VERSION,
            },
            "/parquet-and-dataset-info": {
                "input_type": "dataset",
                "job_runner_version": PROCESSING_STEP_PARQUET_AND_DATASET_INFO_VERSION,
            },
            "config-parquet": {
                "input_type": "config",
                "requires": "/parquet-and-dataset-info",
                "job_runner_version": PROCESSING_STEP_CONFIG_PARQUET_VERSION,
            },
            "split-first-rows-from-parquet": {
                "input_type": "split",
                "requires": "config-parquet",
                "job_runner_version": PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_PARQUET_VERSION,
            },
            "dataset-parquet": {
                "input_type": "dataset",
                "requires": "config-parquet",
                "job_runner_version": PROCESSING_STEP_DATASET_PARQUET_VERSION,
            },
            "config-info": {
                "input_type": "config",
                "requires": "/parquet-and-dataset-info",
                "job_runner_version": PROCESSING_STEP_CONFIG_INFO_VERSION,
            },
            "dataset-info": {
                "input_type": "dataset",
                "requires": "config-info",
                "job_runner_version": PROCESSING_STEP_DATASET_INFO_VERSION,
            },
            "/split-names-from-dataset-info": {
                "input_type": "config",
                "requires": "config-info",
                "job_runner_version": PROCESSING_STEP_SPLIT_NAMES_FROM_DATASET_INFO_VERSION,
            },
            "config-size": {
                "input_type": "config",
                "requires": "/parquet-and-dataset-info",
                "job_runner_version": PROCESSING_STEP_CONFIG_SIZE_VERSION,
            },
            "dataset-size": {
                "input_type": "dataset",
                "requires": "config-size",
                "job_runner_version": PROCESSING_STEP_DATASET_SIZE_VERSION,
            },
            "dataset-split-names-from-streaming": {
                "input_type": "dataset",
                "requires": "/split-names-from-streaming",
                "job_runner_version": PROCESSING_STEP_DATASET_SPLIT_NAMES_FROM_STREAMING_VERSION,
            },
            "dataset-split-names-from-dataset-info": {
                "input_type": "dataset",
                "requires": "/split-names-from-dataset-info",
                "job_runner_version": PROCESSING_STEP_DATASET_SPLIT_NAMES_FROM_DATASET_INFO_VERSION,
            },
        }
    )

    @classmethod
    def from_env(cls) -> "ProcessingGraphConfig":
        # TODO: allow passing the graph via env vars
        return cls()
