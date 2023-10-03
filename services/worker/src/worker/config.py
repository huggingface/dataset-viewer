# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field
from typing import Optional

from environs import Env
from libcommon.config import (
    AssetsConfig,
    CacheConfig,
    CommonConfig,
    LogConfig,
    ParquetMetadataConfig,
    ProcessingGraphConfig,
    QueueConfig,
    RowsIndexConfig,
)

WORKER_CONTENT_MAX_BYTES = 10_000_000
WORKER_DIFFICULTY_MAX = None
WORKER_DIFFICULTY_MIN = None
WORKER_HEARTBEAT_INTERVAL_SECONDS = 60
WORKER_KILL_LONG_JOB_INTERVAL_SECONDS = 60
WORKER_KILL_ZOMBIES_INTERVAL_SECONDS = 10 * 60
WORKER_MAX_DISK_USAGE_PCT = 90
WORKER_MAX_JOB_DURATION_SECONDS = 20 * 60
WORKER_MAX_LOAD_PCT = 70
WORKER_MAX_MEMORY_PCT = 80
WORKER_MAX_MISSING_HEARTBEATS = 5
WORKER_SLEEP_SECONDS = 15
WORKER_STATE_FILE_PATH = None


def get_empty_str_list() -> list[str]:
    return []


@dataclass(frozen=True)
class WorkerConfig:
    content_max_bytes: int = WORKER_CONTENT_MAX_BYTES
    difficulty_max: Optional[int] = WORKER_DIFFICULTY_MAX
    difficulty_min: Optional[int] = WORKER_DIFFICULTY_MIN
    heartbeat_interval_seconds: float = WORKER_HEARTBEAT_INTERVAL_SECONDS
    job_types_blocked: list[str] = field(default_factory=get_empty_str_list)
    job_types_only: list[str] = field(default_factory=get_empty_str_list)
    kill_long_job_interval_seconds: float = WORKER_KILL_LONG_JOB_INTERVAL_SECONDS
    kill_zombies_interval_seconds: float = WORKER_KILL_ZOMBIES_INTERVAL_SECONDS
    max_disk_usage_pct: int = WORKER_MAX_DISK_USAGE_PCT
    max_job_duration_seconds: float = WORKER_MAX_JOB_DURATION_SECONDS
    max_load_pct: int = WORKER_MAX_LOAD_PCT
    max_memory_pct: int = WORKER_MAX_MEMORY_PCT
    max_missing_heartbeats: int = WORKER_MAX_MISSING_HEARTBEATS
    sleep_seconds: float = WORKER_SLEEP_SECONDS
    state_file_path: Optional[str] = WORKER_STATE_FILE_PATH
    storage_paths: list[str] = field(default_factory=get_empty_str_list)

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        env = Env(expand_vars=True)
        with env.prefixed("WORKER_"):
            return cls(
                content_max_bytes=env.int(name="CONTENT_MAX_BYTES", default=WORKER_CONTENT_MAX_BYTES),
                difficulty_max=env.int(name="DIFFICULTY_MAX", default=WORKER_DIFFICULTY_MAX),
                difficulty_min=env.int(name="DIFFICULTY_MIN", default=WORKER_DIFFICULTY_MIN),
                heartbeat_interval_seconds=env.float(
                    name="HEARTBEAT_INTERVAL_SECONDS", default=WORKER_HEARTBEAT_INTERVAL_SECONDS
                ),
                job_types_blocked=env.list(name="JOB_TYPES_BLOCKED", default=get_empty_str_list()),
                job_types_only=env.list(name="JOB_TYPES_ONLY", default=get_empty_str_list()),
                kill_long_job_interval_seconds=env.float(
                    name="KILL_LONG_JOB_INTERVAL_SECONDS", default=WORKER_KILL_LONG_JOB_INTERVAL_SECONDS
                ),
                kill_zombies_interval_seconds=env.float(
                    name="KILL_ZOMBIES_INTERVAL_SECONDS", default=WORKER_KILL_ZOMBIES_INTERVAL_SECONDS
                ),
                max_disk_usage_pct=env.int(name="MAX_DISK_USAGE_PCT", default=WORKER_MAX_DISK_USAGE_PCT),
                max_job_duration_seconds=env.float(
                    name="MAX_JOB_DURATION_SECONDS", default=WORKER_MAX_JOB_DURATION_SECONDS
                ),
                max_load_pct=env.int(name="MAX_LOAD_PCT", default=WORKER_MAX_LOAD_PCT),
                max_memory_pct=env.int(name="MAX_MEMORY_PCT", default=WORKER_MAX_MEMORY_PCT),
                max_missing_heartbeats=env.int(name="MAX_MISSING_HEARTBEATS", default=WORKER_MAX_MISSING_HEARTBEATS),
                sleep_seconds=env.float(name="SLEEP_SECONDS", default=WORKER_SLEEP_SECONDS),
                state_file_path=env.str(
                    name="STATE_FILE_PATH", default=WORKER_STATE_FILE_PATH
                ),  # this environment variable is not expected to be set explicitly, it's set by the worker executor
                storage_paths=env.list(name="STORAGE_PATHS", default=get_empty_str_list()),
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


FIRST_ROWS_CELL_MIN_BYTES = 100
FIRST_ROWS_COLUMNS_MAX_NUMBER = 1_000
FIRST_ROWS_MAX_BYTES = 1_000_000
FIRST_ROWS_MAX_NUMBER = 100
FIRST_ROWS_MIN_NUMBER = 10


@dataclass(frozen=True)
class FirstRowsConfig:
    columns_max_number: int = FIRST_ROWS_COLUMNS_MAX_NUMBER
    max_bytes: int = FIRST_ROWS_MAX_BYTES
    max_number: int = FIRST_ROWS_MAX_NUMBER
    min_cell_bytes: int = FIRST_ROWS_CELL_MIN_BYTES
    min_number: int = FIRST_ROWS_MIN_NUMBER

    @classmethod
    def from_env(cls) -> "FirstRowsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("FIRST_ROWS_"):
            return cls(
                columns_max_number=env.int(name="COLUMNS_MAX_NUMBER", default=FIRST_ROWS_COLUMNS_MAX_NUMBER),
                max_bytes=env.int(name="MAX_BYTES", default=FIRST_ROWS_MAX_BYTES),
                max_number=env.int(name="MAX_NUMBER", default=FIRST_ROWS_MAX_NUMBER),
                min_cell_bytes=env.int(name="CELL_MIN_BYTES", default=FIRST_ROWS_CELL_MIN_BYTES),
                min_number=env.int(name="MIN_NUMBER", default=FIRST_ROWS_MIN_NUMBER),
            )


OPT_IN_OUT_URLS_SCAN_COLUMNS_MAX_NUMBER = 10
OPT_IN_OUT_URLS_SCAN_MAX_CONCURRENT_REQUESTS_NUMBER = 100
OPT_IN_OUT_URLS_SCAN_MAX_REQUESTS_PER_SECOND = 50
OPT_IN_OUT_URLS_SCAN_ROWS_MAX_NUMBER = 100_000
OPT_IN_OUT_URLS_SCAN_SPAWNING_TOKEN = None
OPT_IN_OUT_URLS_SCAN_SPAWNING_URL = "https://opts-api.spawningaiapi.com/api/v2/query/urls"
OPT_IN_OUT_URLS_SCAN_URLS_NUMBER_PER_BATCH = 1000


@dataclass(frozen=True)
class OptInOutUrlsScanConfig:
    columns_max_number: int = FIRST_ROWS_COLUMNS_MAX_NUMBER
    max_concurrent_requests_number: int = OPT_IN_OUT_URLS_SCAN_MAX_CONCURRENT_REQUESTS_NUMBER
    max_requests_per_second: int = OPT_IN_OUT_URLS_SCAN_MAX_REQUESTS_PER_SECOND
    rows_max_number: int = OPT_IN_OUT_URLS_SCAN_ROWS_MAX_NUMBER
    spawning_token: Optional[str] = OPT_IN_OUT_URLS_SCAN_SPAWNING_TOKEN
    spawning_url: str = OPT_IN_OUT_URLS_SCAN_SPAWNING_URL
    urls_number_per_batch: int = OPT_IN_OUT_URLS_SCAN_URLS_NUMBER_PER_BATCH

    @classmethod
    def from_env(cls) -> "OptInOutUrlsScanConfig":
        env = Env(expand_vars=True)
        with env.prefixed("OPT_IN_OUT_URLS_SCAN_"):
            return cls(
                columns_max_number=env.int(name="COLUMNS_MAX_NUMBER", default=OPT_IN_OUT_URLS_SCAN_COLUMNS_MAX_NUMBER),
                max_concurrent_requests_number=env.int(
                    name="MAX_CONCURRENT_REQUESTS_NUMBER", default=OPT_IN_OUT_URLS_SCAN_MAX_CONCURRENT_REQUESTS_NUMBER
                ),
                max_requests_per_second=env.int(
                    name="MAX_REQUESTS_PER_SECOND", default=OPT_IN_OUT_URLS_SCAN_MAX_REQUESTS_PER_SECOND
                ),
                rows_max_number=env.int(name="ROWS_MAX_NUMBER", default=OPT_IN_OUT_URLS_SCAN_ROWS_MAX_NUMBER),
                spawning_token=env.str(name="SPAWNING_TOKEN", default=OPT_IN_OUT_URLS_SCAN_SPAWNING_TOKEN),
                spawning_url=env.str(name="SPAWNING_URL", default=OPT_IN_OUT_URLS_SCAN_SPAWNING_URL),
                urls_number_per_batch=env.int(
                    name="URLS_NUMBER_PER_BATCH", default=OPT_IN_OUT_URLS_SCAN_URLS_NUMBER_PER_BATCH
                ),
            )


PARQUET_AND_INFO_COMMIT_MESSAGE = "Update parquet files"
PARQUET_AND_INFO_COMMITTER_HF_TOKEN = None
PARQUET_AND_INFO_MAX_DATASET_SIZE = 100_000_000
PARQUET_AND_INFO_MAX_EXTERNAL_DATA_FILES = 10_000
PARQUET_AND_INFO_MAX_ROW_GROUP_BYTE_SIZE_FOR_COPY = 100_000_000
PARQUET_AND_INFO_NO_MAX_SIZE_LIMIT_DATASETS: list[str] = []
PARQUET_AND_INFO_SOURCE_REVISION = "main"
PARQUET_AND_INFO_TARGET_REVISION = "refs/convert/parquet"
PARQUET_AND_INFO_URL_TEMPLATE = "/datasets/%s/resolve/%s/%s"


@dataclass(frozen=True)
class ParquetAndInfoConfig:
    commit_message: str = PARQUET_AND_INFO_COMMIT_MESSAGE
    committer_hf_token: Optional[str] = PARQUET_AND_INFO_COMMITTER_HF_TOKEN
    max_dataset_size: int = PARQUET_AND_INFO_MAX_DATASET_SIZE
    max_external_data_files: int = PARQUET_AND_INFO_MAX_EXTERNAL_DATA_FILES
    max_row_group_byte_size_for_copy: int = PARQUET_AND_INFO_MAX_ROW_GROUP_BYTE_SIZE_FOR_COPY
    no_max_size_limit_datasets: list[str] = field(default_factory=PARQUET_AND_INFO_NO_MAX_SIZE_LIMIT_DATASETS.copy)
    source_revision: str = PARQUET_AND_INFO_SOURCE_REVISION
    target_revision: str = PARQUET_AND_INFO_TARGET_REVISION
    url_template: str = PARQUET_AND_INFO_URL_TEMPLATE

    @classmethod
    def from_env(cls) -> "ParquetAndInfoConfig":
        env = Env(expand_vars=True)
        with env.prefixed("PARQUET_AND_INFO_"):
            return cls(
                commit_message=env.str(name="COMMIT_MESSAGE", default=PARQUET_AND_INFO_COMMIT_MESSAGE),
                committer_hf_token=env.str(name="COMMITTER_HF_TOKEN", default=PARQUET_AND_INFO_COMMITTER_HF_TOKEN),
                max_dataset_size=env.int(name="MAX_DATASET_SIZE", default=PARQUET_AND_INFO_MAX_DATASET_SIZE),
                max_external_data_files=env.int(
                    name="MAX_EXTERNAL_DATA_FILES", default=PARQUET_AND_INFO_MAX_EXTERNAL_DATA_FILES
                ),
                max_row_group_byte_size_for_copy=env.int(
                    name="MAX_ROW_GROUP_BYTE_SIZE_FOR_COPY", default=PARQUET_AND_INFO_MAX_ROW_GROUP_BYTE_SIZE_FOR_COPY
                ),
                no_max_size_limit_datasets=env.list(
                    name="NO_MAX_SIZE_LIMIT_DATASETS", default=PARQUET_AND_INFO_NO_MAX_SIZE_LIMIT_DATASETS.copy()
                ),
                source_revision=env.str(name="SOURCE_REVISION", default=PARQUET_AND_INFO_SOURCE_REVISION),
                target_revision=env.str(name="TARGET_REVISION", default=PARQUET_AND_INFO_TARGET_REVISION),
                url_template=env.str(name="URL_TEMPLATE", default=PARQUET_AND_INFO_URL_TEMPLATE),
            )


NUMBA_CACHE_DIR: Optional[str] = None


@dataclass(frozen=True)
class NumbaConfig:
    path: Optional[str] = NUMBA_CACHE_DIR  # not documented

    @classmethod
    def from_env(cls) -> "NumbaConfig":
        env = Env(expand_vars=True)
        with env.prefixed("NUMBA_"):
            return cls(path=env.str(name="CACHE_DIR", default=NUMBA_CACHE_DIR))


CONFIG_NAMES_MAX_NUMBER = 3_000


@dataclass(frozen=True)
class ConfigNamesConfig:
    max_number: int = CONFIG_NAMES_MAX_NUMBER

    @classmethod
    def from_env(cls) -> "ConfigNamesConfig":
        env = Env(expand_vars=True)
        with env.prefixed("CONFIG_NAMES_"):
            return cls(
                max_number=env.int(name="MAX_NUMBER", default=CONFIG_NAMES_MAX_NUMBER),
            )


DUCKDB_INDEX_CACHE_DIRECTORY = None
DUCKDB_INDEX_COMMIT_MESSAGE = "Update duckdb index file"
DUCKDB_INDEX_COMMITTER_HF_TOKEN = None
DUCKDB_INDEX_MAX_PARQUET_SIZE_BYTES = 100_000_000
DUCKDB_INDEX_TARGET_REVISION = "refs/convert/parquet"
DUCKDB_INDEX_URL_TEMPLATE = "/datasets/%s/resolve/%s/%s"
DUCKDB_INDEX_EXTENSIONS_DIRECTORY: Optional[str] = None


@dataclass(frozen=True)
class DuckDbIndexConfig:
    cache_directory: Optional[str] = DUCKDB_INDEX_CACHE_DIRECTORY
    commit_message: str = DUCKDB_INDEX_COMMIT_MESSAGE
    committer_hf_token: Optional[str] = DUCKDB_INDEX_COMMITTER_HF_TOKEN
    target_revision: str = DUCKDB_INDEX_TARGET_REVISION
    url_template: str = DUCKDB_INDEX_URL_TEMPLATE
    max_parquet_size_bytes: int = DUCKDB_INDEX_MAX_PARQUET_SIZE_BYTES
    extensions_directory: Optional[str] = DUCKDB_INDEX_EXTENSIONS_DIRECTORY

    @classmethod
    def from_env(cls) -> "DuckDbIndexConfig":
        env = Env(expand_vars=True)
        with env.prefixed("DUCKDB_INDEX_"):
            return cls(
                cache_directory=env.str(name="CACHE_DIRECTORY", default=DUCKDB_INDEX_CACHE_DIRECTORY),
                commit_message=env.str(name="COMMIT_MESSAGE", default=DUCKDB_INDEX_COMMIT_MESSAGE),
                committer_hf_token=env.str(name="COMMITTER_HF_TOKEN", default=DUCKDB_INDEX_COMMITTER_HF_TOKEN),
                target_revision=env.str(name="TARGET_REVISION", default=DUCKDB_INDEX_TARGET_REVISION),
                url_template=env.str(name="URL_TEMPLATE", default=DUCKDB_INDEX_URL_TEMPLATE),
                max_parquet_size_bytes=env.int(
                    name="MAX_PARQUET_SIZE_BYTES", default=DUCKDB_INDEX_MAX_PARQUET_SIZE_BYTES
                ),
                extensions_directory=env.str(name="EXTENSIONS_DIRECTORY", default=DUCKDB_INDEX_EXTENSIONS_DIRECTORY),
            )


DESCRIPTIVE_STATISTICS_CACHE_DIRECTORY = None
DESCRIPTIVE_STATISTICS_HISTOGRAM_NUM_BINS = 10
DESCRIPTIVE_STATISTICS_MAX_PARQUET_SIZE_BYTES = 100_000_000


@dataclass(frozen=True)
class DescriptiveStatisticsConfig:
    cache_directory: Optional[str] = DESCRIPTIVE_STATISTICS_CACHE_DIRECTORY
    parquet_revision: str = PARQUET_AND_INFO_TARGET_REVISION
    histogram_num_bins: int = DESCRIPTIVE_STATISTICS_HISTOGRAM_NUM_BINS
    max_parquet_size_bytes: int = DESCRIPTIVE_STATISTICS_MAX_PARQUET_SIZE_BYTES

    @classmethod
    def from_env(cls) -> "DescriptiveStatisticsConfig":
        env = Env(expand_vars=True)
        parquet_revision = env.str(name="PARQUET_AND_INFO_TARGET_REVISION", default=PARQUET_AND_INFO_TARGET_REVISION)
        with env.prefixed("DESCRIPTIVE_STATISTICS_"):
            return cls(
                cache_directory=env.str(name="CACHE_DIRECTORY", default=DESCRIPTIVE_STATISTICS_CACHE_DIRECTORY),
                parquet_revision=parquet_revision,
                histogram_num_bins=env.int(
                    name="HISTOGRAM_NUM_BINS",
                    default=DESCRIPTIVE_STATISTICS_HISTOGRAM_NUM_BINS,
                ),
                max_parquet_size_bytes=env.int(
                    name="MAX_PARQUET_SIZE_BYTES", default=DESCRIPTIVE_STATISTICS_MAX_PARQUET_SIZE_BYTES
                ),
            )


@dataclass(frozen=True)
class AppConfig:
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    config_names: ConfigNamesConfig = field(default_factory=ConfigNamesConfig)
    datasets_based: DatasetsBasedConfig = field(default_factory=DatasetsBasedConfig)
    first_rows: FirstRowsConfig = field(default_factory=FirstRowsConfig)
    log: LogConfig = field(default_factory=LogConfig)
    numba: NumbaConfig = field(default_factory=NumbaConfig)
    parquet_and_info: ParquetAndInfoConfig = field(default_factory=ParquetAndInfoConfig)
    processing_graph: ProcessingGraphConfig = field(default_factory=ProcessingGraphConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    rows_index: RowsIndexConfig = field(default_factory=RowsIndexConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    urls_scan: OptInOutUrlsScanConfig = field(default_factory=OptInOutUrlsScanConfig)
    parquet_metadata: ParquetMetadataConfig = field(default_factory=ParquetMetadataConfig)
    duckdb_index: DuckDbIndexConfig = field(default_factory=DuckDbIndexConfig)
    descriptive_statistics: DescriptiveStatisticsConfig = field(default_factory=DescriptiveStatisticsConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            assets=AssetsConfig.from_env(),
            common=CommonConfig.from_env(),
            config_names=ConfigNamesConfig.from_env(),
            cache=CacheConfig.from_env(),
            datasets_based=DatasetsBasedConfig.from_env(),
            first_rows=FirstRowsConfig.from_env(),
            log=LogConfig.from_env(),
            numba=NumbaConfig.from_env(),
            parquet_and_info=ParquetAndInfoConfig.from_env(),
            processing_graph=ProcessingGraphConfig.from_env(),
            queue=QueueConfig.from_env(),
            worker=WorkerConfig.from_env(),
            urls_scan=OptInOutUrlsScanConfig.from_env(),
            parquet_metadata=ParquetMetadataConfig.from_env(),
            duckdb_index=DuckDbIndexConfig.from_env(),
            descriptive_statistics=DescriptiveStatisticsConfig.from_env(),
            rows_index=RowsIndexConfig.from_env(),
        )
