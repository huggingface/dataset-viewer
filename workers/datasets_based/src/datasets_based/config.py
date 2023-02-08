# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import datasets.config
from datasets.utils.logging import log_levels, set_verbosity
from environs import Env
from libcommon.config import (
    AssetsConfig,
    CacheConfig,
    CommonConfig,
    ProcessingGraphConfig,
    QueueConfig,
)

WORKER_LOOP_MAX_DISK_USAGE_PCT = 90
WORKER_LOOP_MAX_LOAD_PCT = 70
WORKER_LOOP_MAX_MEMORY_PCT = 80
WORKER_LOOP_SLEEP_SECONDS = 15


def get_empty_str_list() -> List[str]:
    return []


@dataclass
class WorkerLoopConfig:
    max_disk_usage_pct: int = WORKER_LOOP_MAX_DISK_USAGE_PCT
    max_load_pct: int = WORKER_LOOP_MAX_LOAD_PCT
    max_memory_pct: int = WORKER_LOOP_MAX_MEMORY_PCT
    sleep_seconds: int = WORKER_LOOP_SLEEP_SECONDS
    storage_paths: List[str] = field(default_factory=get_empty_str_list)

    @staticmethod
    def from_env() -> "WorkerLoopConfig":
        env = Env(expand_vars=True)
        with env.prefixed("WORKER_LOOP_"):
            return WorkerLoopConfig(
                max_disk_usage_pct=env.int(name="MAX_DISK_USAGE_PCT", default=WORKER_LOOP_MAX_DISK_USAGE_PCT),
                max_load_pct=env.int(name="MAX_LOAD_PCT", default=WORKER_LOOP_MAX_LOAD_PCT),
                max_memory_pct=env.int(name="MAX_MEMORY_PCT", default=WORKER_LOOP_MAX_MEMORY_PCT),
                sleep_seconds=env.int(name="SLEEP_SECONDS", default=WORKER_LOOP_SLEEP_SECONDS),
                storage_paths=env.list(name="STORAGE_PATHS", default=get_empty_str_list()),
            )


DATASETS_BASED_ENDPOINT = "/config-names"
DATASETS_BASED_HF_DATASETS_CACHE = None


@dataclass
class DatasetsBasedConfig:
    endpoint: str = DATASETS_BASED_ENDPOINT
    _hf_datasets_cache: Union[str, Path, None] = DATASETS_BASED_HF_DATASETS_CACHE

    def __post_init__(self) -> None:
        self.hf_datasets_cache = (
            datasets.config.HF_DATASETS_CACHE if self._hf_datasets_cache is None else Path(self._hf_datasets_cache)
        )

    @staticmethod
    def from_env() -> "DatasetsBasedConfig":
        env = Env(expand_vars=True)
        with env.prefixed("DATASETS_BASED_"):
            return DatasetsBasedConfig(
                endpoint=env.str(name="ENDPOINT", default=DATASETS_BASED_ENDPOINT),
                _hf_datasets_cache=env.str(name="HF_DATASETS_CACHE", default=DATASETS_BASED_HF_DATASETS_CACHE),
            )


FIRST_ROWS_MAX_BYTES = 1_000_000
FIRST_ROWS_MAX_NUMBER = 100
FIRST_ROWS_CELL_MIN_BYTES = 100
FIRST_ROWS_MIN_NUMBER = 10
FIRST_ROWS_COLUMNS_MAX_NUMBER = 1_000


@dataclass
class FirstRowsConfig:
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    max_bytes: int = FIRST_ROWS_MAX_BYTES
    max_number: int = FIRST_ROWS_MAX_NUMBER
    min_cell_bytes: int = FIRST_ROWS_CELL_MIN_BYTES
    min_number: int = FIRST_ROWS_MIN_NUMBER
    columns_max_number: int = FIRST_ROWS_COLUMNS_MAX_NUMBER

    @staticmethod
    def from_env() -> "FirstRowsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("FIRST_ROWS_"):
            return FirstRowsConfig(
                assets=AssetsConfig.from_env(),
                max_bytes=env.int(name="MAX_BYTES", default=FIRST_ROWS_MAX_BYTES),
                max_number=env.int(name="MAX_NUMBER", default=FIRST_ROWS_MAX_NUMBER),
                min_cell_bytes=env.int(name="CELL_MIN_BYTES", default=FIRST_ROWS_CELL_MIN_BYTES),
                min_number=env.int(name="MIN_NUMBER", default=FIRST_ROWS_MIN_NUMBER),
                columns_max_number=env.int(name="COLUMNS_MAX_NUMBER", default=FIRST_ROWS_COLUMNS_MAX_NUMBER),
            )


PARQUET_AND_DATASET_INFO_COMMIT_MESSAGE = "Update parquet files"
PARQUET_AND_DATASET_INFO_COMMITTER_HF_TOKEN = None
PARQUET_AND_DATASET_INFO_MAX_DATASET_SIZE = 100_000_000
PARQUET_AND_DATASET_INFO_SOURCE_REVISION = "main"
PARQUET_AND_DATASET_INFO_TARGET_REVISION = "refs/convert/parquet"
PARQUET_AND_DATASET_INFO_URL_TEMPLATE = "/datasets/%s/resolve/%s/%s"


@dataclass
class ParquetAndDatasetInfoConfig:
    blocked_datasets: List[str] = field(default_factory=get_empty_str_list)
    supported_datasets: List[str] = field(default_factory=get_empty_str_list)
    commit_message: str = PARQUET_AND_DATASET_INFO_COMMIT_MESSAGE
    committer_hf_token: Optional[str] = PARQUET_AND_DATASET_INFO_COMMITTER_HF_TOKEN
    max_dataset_size: int = PARQUET_AND_DATASET_INFO_MAX_DATASET_SIZE
    source_revision: str = PARQUET_AND_DATASET_INFO_SOURCE_REVISION
    target_revision: str = PARQUET_AND_DATASET_INFO_TARGET_REVISION
    url_template: str = PARQUET_AND_DATASET_INFO_URL_TEMPLATE

    @staticmethod
    def from_env() -> "ParquetAndDatasetInfoConfig":
        env = Env(expand_vars=True)
        with env.prefixed("PARQUET_AND_DATASET_INFO_"):
            return ParquetAndDatasetInfoConfig(
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


@dataclass
class AppConfig:
    cache: CacheConfig = field(default_factory=CacheConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    datasets_based: DatasetsBasedConfig = field(default_factory=DatasetsBasedConfig)
    processing_graph: ProcessingGraphConfig = field(default_factory=ProcessingGraphConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    worker_loop: WorkerLoopConfig = field(default_factory=WorkerLoopConfig)

    def __post_init__(self):
        # Ensure the datasets library uses the expected HuggingFace endpoint
        datasets.config.HF_ENDPOINT = self.common.hf_endpoint
        # Don't increase the datasets download counts on huggingface.co
        datasets.config.HF_UPDATE_DOWNLOAD_COUNTS = False
        # Set logs from the datasets library to the least verbose
        set_verbosity(log_levels["critical"])

        # Note: self.common.hf_endpoint is ignored by the huggingface_hub library for now (see
        # the discussion at https://github.com/huggingface/datasets/pull/5196), and this breaks
        # various of the datasets functions. The fix, for now, is to set the HF_ENDPOINT
        # environment variable to the desired value.

        # Add the datasets and numba cache paths to the list of storage paths, to ensure the disk is not full
        env = Env(expand_vars=True)
        numba_path = env.str(name="NUMBA_CACHE_DIR", default=None)
        additional_paths = {str(self.datasets_based.hf_datasets_cache), str(datasets.config.HF_MODULES_CACHE)}
        if numba_path:
            additional_paths.add(numba_path)
        self.worker_loop.storage_paths = list(set(self.worker_loop.storage_paths).union(additional_paths))

    @staticmethod
    def from_env() -> "AppConfig":
        return AppConfig(
            # First process the common configuration to setup the logging
            common=CommonConfig.from_env(),
            cache=CacheConfig.from_env(),
            datasets_based=DatasetsBasedConfig.from_env(),
            processing_graph=ProcessingGraphConfig.from_env(),
            queue=QueueConfig.from_env(),
            worker_loop=WorkerLoopConfig.from_env(),
        )
