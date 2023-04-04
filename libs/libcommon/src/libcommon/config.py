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
ASSETS_STORE_DIRECTORY = None


@dataclass(frozen=True)
class AssetsConfig:
    base_url: str = ASSETS_BASE_URL
    storage_directory: Optional[str] = ASSETS_STORE_DIRECTORY

    @classmethod
    def from_env(cls) -> "AssetsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("ASSETS_"):
            return cls(
                base_url=env.str(name="BASE_URL", default=ASSETS_BASE_URL),
                storage_directory=env.str(name="STORAGE_DIRECTORY", default=ASSETS_STORE_DIRECTORY),
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
                "requires": "/split-names-from-streaming",
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
