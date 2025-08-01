# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field
from typing import Optional

from environs import Env
from libcommon.config import (
    AssetsConfig,
    CacheConfig,
    CommitterConfig,
    CommonConfig,
    LogConfig,
    ParquetMetadataConfig,
    QueueConfig,
    RowsIndexConfig,
    S3Config,
)

WORKER_UVICORN_HOSTNAME = "localhost"
WORKER_UVICORN_NUM_WORKERS = 2
WORKER_UVICORN_PORT = 8000


@dataclass(frozen=True)
class UvicornConfig:
    hostname: str = WORKER_UVICORN_HOSTNAME
    num_workers: int = WORKER_UVICORN_NUM_WORKERS
    port: int = WORKER_UVICORN_PORT

    @classmethod
    def from_env(cls) -> "UvicornConfig":
        env = Env(expand_vars=True)
        with env.prefixed("WORKER_UVICORN_"):
            return cls(
                hostname=env.str(name="HOSTNAME", default=WORKER_UVICORN_HOSTNAME),
                num_workers=env.int(name="NUM_WORKERS", default=WORKER_UVICORN_NUM_WORKERS),
                port=env.int(name="PORT", default=WORKER_UVICORN_PORT),
            )


WORKER_CONTENT_MAX_BYTES = 10_000_000
WORKER_DIFFICULTY_MAX = None
WORKER_DIFFICULTY_MIN = None
WORKER_HEARTBEAT_INTERVAL_SECONDS = 60
WORKER_KILL_LONG_JOB_INTERVAL_SECONDS = 60
WORKER_KILL_ZOMBIES_INTERVAL_SECONDS = 10 * 60
WORKER_MAX_JOB_DURATION_SECONDS = 20 * 60
WORKER_MAX_LOAD_PCT = 70
WORKER_MAX_MEMORY_PCT = 80
WORKER_MAX_MISSING_HEARTBEATS = 5
WORKER_SLEEP_SECONDS = 15
WORKER_STATE_FILE_PATH = None


@dataclass(frozen=True)
class WorkerConfig:
    content_max_bytes: int = WORKER_CONTENT_MAX_BYTES
    difficulty_max: Optional[int] = WORKER_DIFFICULTY_MAX
    difficulty_min: Optional[int] = WORKER_DIFFICULTY_MIN
    heartbeat_interval_seconds: float = WORKER_HEARTBEAT_INTERVAL_SECONDS
    kill_long_job_interval_seconds: float = WORKER_KILL_LONG_JOB_INTERVAL_SECONDS
    kill_zombies_interval_seconds: float = WORKER_KILL_ZOMBIES_INTERVAL_SECONDS
    max_job_duration_seconds: float = WORKER_MAX_JOB_DURATION_SECONDS
    max_load_pct: int = WORKER_MAX_LOAD_PCT
    max_memory_pct: int = WORKER_MAX_MEMORY_PCT
    max_missing_heartbeats: int = WORKER_MAX_MISSING_HEARTBEATS
    sleep_seconds: float = WORKER_SLEEP_SECONDS
    state_file_path: Optional[str] = WORKER_STATE_FILE_PATH

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        env = Env(expand_vars=True)
        with env.prefixed("WORKER_"):
            return cls(
                content_max_bytes=env.int(name="CONTENT_MAX_BYTES", default=WORKER_CONTENT_MAX_BYTES),
                difficulty_max=env.int(name="DIFFICULTY_MAX", default=WORKER_DIFFICULTY_MAX),
                difficulty_min=env.int(name="DIFFICULTY_MIN", default=WORKER_DIFFICULTY_MIN),
                heartbeat_interval_seconds=env.float(
                    name="HEARTBEAT_INTERVAL_SECONDS",
                    default=WORKER_HEARTBEAT_INTERVAL_SECONDS,
                ),
                kill_long_job_interval_seconds=env.float(
                    name="KILL_LONG_JOB_INTERVAL_SECONDS",
                    default=WORKER_KILL_LONG_JOB_INTERVAL_SECONDS,
                ),
                kill_zombies_interval_seconds=env.float(
                    name="KILL_ZOMBIES_INTERVAL_SECONDS",
                    default=WORKER_KILL_ZOMBIES_INTERVAL_SECONDS,
                ),
                max_job_duration_seconds=env.float(
                    name="MAX_JOB_DURATION_SECONDS",
                    default=WORKER_MAX_JOB_DURATION_SECONDS,
                ),
                max_load_pct=env.int(name="MAX_LOAD_PCT", default=WORKER_MAX_LOAD_PCT),
                max_memory_pct=env.int(name="MAX_MEMORY_PCT", default=WORKER_MAX_MEMORY_PCT),
                max_missing_heartbeats=env.int(name="MAX_MISSING_HEARTBEATS", default=WORKER_MAX_MISSING_HEARTBEATS),
                sleep_seconds=env.float(name="SLEEP_SECONDS", default=WORKER_SLEEP_SECONDS),
                state_file_path=env.str(
                    name="STATE_FILE_PATH", default=WORKER_STATE_FILE_PATH
                ),  # this environment variable is not expected to be set explicitly, it's set by the worker executor
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


FIRST_ROWS_MIN_CELL_BYTES = 100
FIRST_ROWS_COLUMNS_MAX_NUMBER = 1_000
FIRST_ROWS_MAX_BYTES = 1_000_000
FIRST_ROWS_MIN_NUMBER = 10


@dataclass(frozen=True)
class FirstRowsConfig:
    columns_max_number: int = FIRST_ROWS_COLUMNS_MAX_NUMBER
    max_bytes: int = FIRST_ROWS_MAX_BYTES
    min_cell_bytes: int = FIRST_ROWS_MIN_CELL_BYTES
    min_number: int = FIRST_ROWS_MIN_NUMBER

    @classmethod
    def from_env(cls) -> "FirstRowsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("FIRST_ROWS_"):
            return cls(
                columns_max_number=env.int(name="COLUMNS_MAX_NUMBER", default=FIRST_ROWS_COLUMNS_MAX_NUMBER),
                max_bytes=env.int(name="MAX_BYTES", default=FIRST_ROWS_MAX_BYTES),
                min_cell_bytes=env.int(name="MIN_CELL_BYTES", default=FIRST_ROWS_MIN_CELL_BYTES),
                min_number=env.int(name="MIN_NUMBER", default=FIRST_ROWS_MIN_NUMBER),
            )


OPT_IN_OUT_URLS_SCAN_COLUMNS_MAX_NUMBER = 10
OPT_IN_OUT_URLS_SCAN_MAX_CONCURRENT_REQUESTS_NUMBER = 50
OPT_IN_OUT_URLS_SCAN_MAX_REQUESTS_PER_SECOND = 25
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
                columns_max_number=env.int(
                    name="COLUMNS_MAX_NUMBER",
                    default=OPT_IN_OUT_URLS_SCAN_COLUMNS_MAX_NUMBER,
                ),
                max_concurrent_requests_number=env.int(
                    name="MAX_CONCURRENT_REQUESTS_NUMBER",
                    default=OPT_IN_OUT_URLS_SCAN_MAX_CONCURRENT_REQUESTS_NUMBER,
                ),
                max_requests_per_second=env.int(
                    name="MAX_REQUESTS_PER_SECOND",
                    default=OPT_IN_OUT_URLS_SCAN_MAX_REQUESTS_PER_SECOND,
                ),
                rows_max_number=env.int(name="ROWS_MAX_NUMBER", default=OPT_IN_OUT_URLS_SCAN_ROWS_MAX_NUMBER),
                spawning_token=env.str(name="SPAWNING_TOKEN", default=OPT_IN_OUT_URLS_SCAN_SPAWNING_TOKEN),
                spawning_url=env.str(name="SPAWNING_URL", default=OPT_IN_OUT_URLS_SCAN_SPAWNING_URL),
                urls_number_per_batch=env.int(
                    name="URLS_NUMBER_PER_BATCH",
                    default=OPT_IN_OUT_URLS_SCAN_URLS_NUMBER_PER_BATCH,
                ),
            )


PRESIDIO_ENTITIES_SCAN_COLUMNS_MAX_NUMBER = 10
PRESIDIO_ENTITIES_SCAN_MAX_TEXT_LENGTH = 1000
PRESIDIO_ENTITIES_SCAN_ROWS_MAX_NUMBER = 10_000


@dataclass(frozen=True)
class PresidioEntitiesScanConfig:
    columns_max_number: int = PRESIDIO_ENTITIES_SCAN_COLUMNS_MAX_NUMBER
    max_text_length: int = PRESIDIO_ENTITIES_SCAN_MAX_TEXT_LENGTH
    rows_max_number: int = PRESIDIO_ENTITIES_SCAN_ROWS_MAX_NUMBER

    @classmethod
    def from_env(cls) -> "PresidioEntitiesScanConfig":
        env = Env(expand_vars=True)
        with env.prefixed("PRESIDIO_ENTITIES_SCAN_"):
            return cls(
                columns_max_number=env.int(
                    name="COLUMNS_MAX_NUMBER",
                    default=PRESIDIO_ENTITIES_SCAN_COLUMNS_MAX_NUMBER,
                ),
                max_text_length=env.int(
                    name="MAX_TEXT_LENGTH",
                    default=PRESIDIO_ENTITIES_SCAN_MAX_TEXT_LENGTH,
                ),
                rows_max_number=env.int(
                    name="ROWS_MAX_NUMBER",
                    default=PRESIDIO_ENTITIES_SCAN_ROWS_MAX_NUMBER,
                ),
            )


PARQUET_AND_INFO_COMMIT_MESSAGE = "Update parquet files"
PARQUET_AND_INFO_COMMITTER_HF_TOKEN = None
PARQUET_AND_INFO_MAX_DATASET_SIZE_BYTES = 100_000_000
PARQUET_AND_INFO_MAX_ROW_GROUP_BYTE_SIZE_FOR_COPY = 100_000_000
PARQUET_AND_INFO_SOURCE_REVISION = "main"
PARQUET_AND_INFO_TARGET_REVISION = "refs/convert/parquet"
PARQUET_AND_INFO_URL_TEMPLATE = "/datasets/%s/resolve/%s/%s"
PARQUET_AND_INFO_FULLY_CONVERTED_DATASETS: list[str] = []


@dataclass(frozen=True)
class ParquetAndInfoConfig:
    commit_message: str = PARQUET_AND_INFO_COMMIT_MESSAGE
    max_dataset_size_bytes: int = PARQUET_AND_INFO_MAX_DATASET_SIZE_BYTES
    max_row_group_byte_size_for_copy: int = PARQUET_AND_INFO_MAX_ROW_GROUP_BYTE_SIZE_FOR_COPY
    source_revision: str = PARQUET_AND_INFO_SOURCE_REVISION
    target_revision: str = PARQUET_AND_INFO_TARGET_REVISION
    url_template: str = PARQUET_AND_INFO_URL_TEMPLATE
    fully_converted_datasets: list[str] = field(default_factory=PARQUET_AND_INFO_FULLY_CONVERTED_DATASETS.copy)

    @classmethod
    def from_env(cls) -> "ParquetAndInfoConfig":
        env = Env(expand_vars=True)
        with env.prefixed("PARQUET_AND_INFO_"):
            return cls(
                commit_message=env.str(name="COMMIT_MESSAGE", default=PARQUET_AND_INFO_COMMIT_MESSAGE),
                max_dataset_size_bytes=env.int(
                    name="MAX_DATASET_SIZE_BYTES",
                    default=PARQUET_AND_INFO_MAX_DATASET_SIZE_BYTES,
                ),
                max_row_group_byte_size_for_copy=env.int(
                    name="MAX_ROW_GROUP_BYTE_SIZE_FOR_COPY",
                    default=PARQUET_AND_INFO_MAX_ROW_GROUP_BYTE_SIZE_FOR_COPY,
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


CONFIG_NAMES_MAX_NUMBER = 4_000


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


SPLIT_NAMES_MAX_NUMBER = 30


@dataclass(frozen=True)
class SplitNamesConfig:
    max_number: int = SPLIT_NAMES_MAX_NUMBER

    @classmethod
    def from_env(cls) -> "SplitNamesConfig":
        env = Env(expand_vars=True)
        with env.prefixed("SPLIT_NAMES_"):
            return cls(
                max_number=env.int(name="MAX_NUMBER", default=SPLIT_NAMES_MAX_NUMBER),
            )


DESCRIPTIVE_STATISTICS_CACHE_DIRECTORY = None
DESCRIPTIVE_STATISTICS_MAX_SPLIT_SIZE_BYTES = 100_000_000


@dataclass(frozen=True)
class DescriptiveStatisticsConfig:
    cache_directory: Optional[str] = DESCRIPTIVE_STATISTICS_CACHE_DIRECTORY
    parquet_revision: str = PARQUET_AND_INFO_TARGET_REVISION
    max_split_size_bytes: int = DESCRIPTIVE_STATISTICS_MAX_SPLIT_SIZE_BYTES

    @classmethod
    def from_env(cls) -> "DescriptiveStatisticsConfig":
        env = Env(expand_vars=True)
        parquet_revision = env.str(
            name="PARQUET_AND_INFO_TARGET_REVISION",
            default=PARQUET_AND_INFO_TARGET_REVISION,
        )
        with env.prefixed("DESCRIPTIVE_STATISTICS_"):
            return cls(
                cache_directory=env.str(
                    name="CACHE_DIRECTORY",
                    default=DESCRIPTIVE_STATISTICS_CACHE_DIRECTORY,
                ),
                parquet_revision=parquet_revision,
                max_split_size_bytes=env.int(
                    name="MAX_SPLIT_SIZE_BYTES",
                    default=DESCRIPTIVE_STATISTICS_MAX_SPLIT_SIZE_BYTES,
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
    queue: QueueConfig = field(default_factory=QueueConfig)
    rows_index: RowsIndexConfig = field(default_factory=RowsIndexConfig)
    s3: S3Config = field(default_factory=S3Config)
    split_names: SplitNamesConfig = field(default_factory=SplitNamesConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    urls_scan: OptInOutUrlsScanConfig = field(default_factory=OptInOutUrlsScanConfig)
    presidio_scan: PresidioEntitiesScanConfig = field(default_factory=PresidioEntitiesScanConfig)
    parquet_metadata: ParquetMetadataConfig = field(default_factory=ParquetMetadataConfig)
    descriptive_statistics: DescriptiveStatisticsConfig = field(default_factory=DescriptiveStatisticsConfig)
    committer: CommitterConfig = field(default_factory=CommitterConfig)

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
            queue=QueueConfig.from_env(),
            rows_index=RowsIndexConfig.from_env(),
            s3=S3Config.from_env(),
            split_names=SplitNamesConfig.from_env(),
            worker=WorkerConfig.from_env(),
            urls_scan=OptInOutUrlsScanConfig.from_env(),
            presidio_scan=PresidioEntitiesScanConfig.from_env(),
            parquet_metadata=ParquetMetadataConfig.from_env(),
            descriptive_statistics=DescriptiveStatisticsConfig.from_env(),
            committer=CommitterConfig.from_env(),
        )
