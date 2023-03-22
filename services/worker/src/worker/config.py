# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field
from typing import List, Optional

from environs import Env
from libcommon.config import (
    AssetsConfig,
    CacheConfig,
    CommonConfig,
    ProcessingGraphConfig,
    QueueConfig,
)

WORKER_CONTENT_MAX_BYTES = 10_000_000
WORKER_ENDPOINT = "/config-names"
WORKER_HEARTBEAT_INTERVAL_SECONDS = 60
WORKER_MAX_JOB_DURATION_SECONDS = 20 * 60
WORKER_KILL_LONG_JOB_INTERVAL_SECONDS = 60
WORKER_KILL_ZOMBIES_INTERVAL_SECONDS = 10 * 60
WORKER_MAX_DISK_USAGE_PCT = 90
WORKER_MAX_LOAD_PCT = 70
WORKER_MAX_MEMORY_PCT = 80
WORKER_MAX_MISSING_HEARTBEATS = 5
WORKER_SLEEP_SECONDS = 15
WORKER_STATE_FILE_PATH = None


def get_empty_str_list() -> List[str]:
    return []


@dataclass(frozen=True)
class WorkerConfig:
    content_max_bytes: int = WORKER_CONTENT_MAX_BYTES
    max_disk_usage_pct: int = WORKER_MAX_DISK_USAGE_PCT
    max_load_pct: int = WORKER_MAX_LOAD_PCT
    max_memory_pct: int = WORKER_MAX_MEMORY_PCT
    only_job_types: list[str] = field(default_factory=get_empty_str_list)
    sleep_seconds: int = WORKER_SLEEP_SECONDS
    storage_paths: List[str] = field(default_factory=get_empty_str_list)
    state_file_path: Optional[str] = WORKER_STATE_FILE_PATH
    heartbeat_interval_seconds: int = WORKER_HEARTBEAT_INTERVAL_SECONDS
    max_missing_heartbeats: int = WORKER_MAX_MISSING_HEARTBEATS
    kill_zombies_interval_seconds: int = WORKER_KILL_ZOMBIES_INTERVAL_SECONDS
    max_job_duration_seconds: int = WORKER_MAX_JOB_DURATION_SECONDS
    kill_long_job_interval_seconds: int = WORKER_KILL_LONG_JOB_INTERVAL_SECONDS

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        env = Env(expand_vars=True)
        with env.prefixed("WORKER_"):
            return cls(
                content_max_bytes=env.int(name="CONTENT_MAX_BYTES", default=WORKER_CONTENT_MAX_BYTES),
                max_disk_usage_pct=env.int(name="MAX_DISK_USAGE_PCT", default=WORKER_MAX_DISK_USAGE_PCT),
                max_load_pct=env.int(name="MAX_LOAD_PCT", default=WORKER_MAX_LOAD_PCT),
                max_memory_pct=env.int(name="MAX_MEMORY_PCT", default=WORKER_MAX_MEMORY_PCT),
                sleep_seconds=env.int(name="SLEEP_SECONDS", default=WORKER_SLEEP_SECONDS),
                only_job_types=env.list(name="ONLY_JOB_TYPES", default=get_empty_str_list()),
                storage_paths=env.list(name="STORAGE_PATHS", default=get_empty_str_list()),
                state_file_path=env.str(
                    name="STATE_FILE_PATH", default=WORKER_STATE_FILE_PATH
                ),  # this environment variable is not expected to be set explicitly, it's set by the worker executor
                heartbeat_interval_seconds=env.int(
                    name="HEARTBEAT_INTERVAL_SECONDS", default=WORKER_HEARTBEAT_INTERVAL_SECONDS
                ),
                max_missing_heartbeats=env.int(name="MAX_MISSING_HEARTBEATS", default=WORKER_MAX_MISSING_HEARTBEATS),
                kill_zombies_interval_seconds=env.int(
                    name="KILL_ZOMBIES_INTERVAL_SECONDS", default=WORKER_KILL_ZOMBIES_INTERVAL_SECONDS
                ),
                max_job_duration_seconds=env.int(
                    name="MAX_JOB_DURATION_SECONDS", default=WORKER_MAX_JOB_DURATION_SECONDS
                ),
                kill_long_job_interval_seconds=env.int(
                    name="KILL_LONG_JOB_INTERVAL_SECONDS", default=WORKER_KILL_LONG_JOB_INTERVAL_SECONDS
                ),
            )


DATASETS_BASED_HF_DATASETS_CACHE = None


@dataclass(frozen=True)
class DatasetsBasedConfig:
    hf_datasets_cache: Optional[str] = DATASETS_BASED_HF_DATASETS_CACHE

    @classmethod
    def from_env(cls) -> "DatasetsBasedConfig":
        env = Env(expand_vars=True)
        with env.prefixed("DATASETS_BASED_"):
            return cls(
                hf_datasets_cache=env.str(name="HF_DATASETS_CACHE", default=DATASETS_BASED_HF_DATASETS_CACHE),
            )


FIRST_ROWS_MAX_BYTES = 1_000_000
FIRST_ROWS_MAX_NUMBER = 100
FIRST_ROWS_CELL_MIN_BYTES = 100
FIRST_ROWS_MIN_NUMBER = 10
FIRST_ROWS_COLUMNS_MAX_NUMBER = 1_000


@dataclass(frozen=True)
class FirstRowsConfig:
    max_bytes: int = FIRST_ROWS_MAX_BYTES
    max_number: int = FIRST_ROWS_MAX_NUMBER
    min_cell_bytes: int = FIRST_ROWS_CELL_MIN_BYTES
    min_number: int = FIRST_ROWS_MIN_NUMBER
    columns_max_number: int = FIRST_ROWS_COLUMNS_MAX_NUMBER

    @classmethod
    def from_env(cls) -> "FirstRowsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("FIRST_ROWS_"):
            return cls(
                max_bytes=env.int(name="MAX_BYTES", default=FIRST_ROWS_MAX_BYTES),
                max_number=env.int(name="MAX_NUMBER", default=FIRST_ROWS_MAX_NUMBER),
                min_cell_bytes=env.int(name="CELL_MIN_BYTES", default=FIRST_ROWS_CELL_MIN_BYTES),
                min_number=env.int(name="MIN_NUMBER", default=FIRST_ROWS_MIN_NUMBER),
                columns_max_number=env.int(name="COLUMNS_MAX_NUMBER", default=FIRST_ROWS_COLUMNS_MAX_NUMBER),
            )


PARQUET_AND_DATASET_INFO_COMMIT_MESSAGE = "Update parquet files"
PARQUET_AND_DATASET_INFO_COMMITTER_HF_TOKEN = None
PARQUET_AND_DATASET_INFO_MAX_DATASET_SIZE = 100_000_000
PARQUET_AND_DATASET_INFO_MAX_EXTERNAL_DATA_FILES = 10_000
PARQUET_AND_DATASET_INFO_SOURCE_REVISION = "main"
PARQUET_AND_DATASET_INFO_TARGET_REVISION = "refs/convert/parquet"
PARQUET_AND_DATASET_INFO_URL_TEMPLATE = "/datasets/%s/resolve/%s/%s"


@dataclass(frozen=True)
class ParquetAndDatasetInfoConfig:
    blocked_datasets: List[str] = field(default_factory=get_empty_str_list)
    supported_datasets: List[str] = field(default_factory=get_empty_str_list)
    commit_message: str = PARQUET_AND_DATASET_INFO_COMMIT_MESSAGE
    committer_hf_token: Optional[str] = PARQUET_AND_DATASET_INFO_COMMITTER_HF_TOKEN
    max_dataset_size: int = PARQUET_AND_DATASET_INFO_MAX_DATASET_SIZE
    max_external_data_files: int = PARQUET_AND_DATASET_INFO_MAX_EXTERNAL_DATA_FILES
    source_revision: str = PARQUET_AND_DATASET_INFO_SOURCE_REVISION
    target_revision: str = PARQUET_AND_DATASET_INFO_TARGET_REVISION
    url_template: str = PARQUET_AND_DATASET_INFO_URL_TEMPLATE

    @classmethod
    def from_env(cls) -> "ParquetAndDatasetInfoConfig":
        env = Env(expand_vars=True)
        with env.prefixed("PARQUET_AND_DATASET_INFO_"):
            return cls(
                blocked_datasets=env.list(name="BLOCKED_DATASETS", default=get_empty_str_list()),
                supported_datasets=env.list(name="SUPPORTED_DATASETS", default=get_empty_str_list()),
                commit_message=env.str(name="COMMIT_MESSAGE", default=PARQUET_AND_DATASET_INFO_COMMIT_MESSAGE),
                committer_hf_token=env.str(
                    name="COMMITTER_HF_TOKEN", default=PARQUET_AND_DATASET_INFO_COMMITTER_HF_TOKEN
                ),
                max_dataset_size=env.int(name="MAX_DATASET_SIZE", default=PARQUET_AND_DATASET_INFO_MAX_DATASET_SIZE),
                source_revision=env.str(name="SOURCE_REVISION", default=PARQUET_AND_DATASET_INFO_SOURCE_REVISION),
                target_revision=env.str(name="TARGET_REVISION", default=PARQUET_AND_DATASET_INFO_TARGET_REVISION),
                url_template=env.str(name="URL_TEMPLATE", default=PARQUET_AND_DATASET_INFO_URL_TEMPLATE),
            )


NUMBA_CACHE_DIR: Optional[str] = None


@dataclass(frozen=True)
class NumbaConfig:
    path: Optional[str] = NUMBA_CACHE_DIR  # not documented

    @classmethod
    def from_env(cls) -> "NumbaConfig":
        env = Env(expand_vars=True)
        with env.prefixed("NUMBA_"):
            return cls(path=env.str(name="NUMBA_CACHE_DIR", default=NUMBA_CACHE_DIR))


@dataclass(frozen=True)
class AppConfig:
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    datasets_based: DatasetsBasedConfig = field(default_factory=DatasetsBasedConfig)
    numba: NumbaConfig = field(default_factory=NumbaConfig)
    processing_graph: ProcessingGraphConfig = field(default_factory=ProcessingGraphConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            assets=AssetsConfig.from_env(),
            common=CommonConfig.from_env(),
            cache=CacheConfig.from_env(),
            datasets_based=DatasetsBasedConfig.from_env(),
            numba=NumbaConfig.from_env(),
            processing_graph=ProcessingGraphConfig.from_env(),
            queue=QueueConfig.from_env(),
            worker=WorkerConfig.from_env(),
        )
