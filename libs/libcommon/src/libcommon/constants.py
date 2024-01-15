# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

CACHE_COLLECTION_RESPONSES = "cachedResponsesBlue"
CACHE_MONGOENGINE_ALIAS = "cache"
HF_DATASETS_CACHE_APPNAME = "hf_datasets_cache"
PARQUET_METADATA_CACHE_APPNAME = "datasets_server_parquet_metadata"
DESCRIPTIVE_STATISTICS_CACHE_APPNAME = "datasets_server_descriptive_statistics"
DUCKDB_INDEX_CACHE_APPNAME = "datasets_server_duckdb_index"
DUCKDB_INDEX_DOWNLOADS_SUBDIRECTORY = "downloads"
DUCKDB_INDEX_JOB_RUNNER_SUBDIRECTORY = "job_runner"
CACHE_METRICS_COLLECTION = "cacheTotalMetric"
QUEUE_METRICS_COLLECTION = "jobTotalMetric"
METRICS_MONGOENGINE_ALIAS = "metrics"
QUEUE_COLLECTION_JOBS = "jobsBlue"
QUEUE_COLLECTION_LOCKS = "locks"
QUEUE_MONGOENGINE_ALIAS = "queue"
QUEUE_TTL_SECONDS = 600  # 10 minutes
LOCK_TTL_SECONDS_NO_OWNER = 600  # 10 minutes
LOCK_TTL_SECONDS_TO_START_JOB = 600  # 10 minutes
LOCK_TTL_SECONDS_TO_WRITE_ON_GIT_BRANCH = 3600  # 1 hour

MAX_FAILED_RUNS = 3
DATASET_SEPARATOR = "--"
DEFAULT_DIFFICULTY = 50
DEFAULT_DIFFICULTY_MAX = 100
DEFAULT_DIFFICULTY_MIN = 0
DEFAULT_INPUT_TYPE = "dataset"
DEFAULT_JOB_RUNNER_VERSION = 1
DIFFICULTY_BONUS_BY_FAILED_RUNS = 20
MIN_BYTES_FOR_BONUS_DIFFICULTY = 3_000_000_000

PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_ROW_GROUP_SIZE_FOR_AUDIO_DATASETS = 100
PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_ROW_GROUP_SIZE_FOR_IMAGE_DATASETS = 100
PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_ROW_GROUP_SIZE_FOR_BINARY_DATASETS = 100
PARQUET_REVISION = "refs/convert/parquet"

ERROR_CODES_TO_RETRY = {
    "CreateCommitError",
    "ExternalServerError",
    "JobManagerCrashedError",
    "LockedDatasetTimeoutError",
    "StreamingRowsError",
}

EXTERNAL_DATASET_SCRIPT_PATTERN = "datasets_modules/datasets"

# Arrays are not immutable, we have to take care of not modifying them
# Anyway: in all this file, we allow constant reassignment (no use of Final)
CONFIG_HAS_VIEWER_KINDS = ["config-size"]
CONFIG_INFO_KINDS = ["config-info"]
CONFIG_PARQUET_METADATA_KINDS = ["config-parquet-metadata"]
CONFIG_PARQUET_AND_METADATA_KINDS = ["config-parquet", "config-parquet-metadata"]
CONFIG_SPLIT_NAMES_KINDS = ["config-split-names-from-info", "config-split-names-from-streaming"]
DATASET_CONFIG_NAMES_KINDS = ["dataset-config-names"]
DATASET_INFO_KINDS = ["dataset-info"]
SPLIT_DUCKDB_INDEX_KINDS = ["split-duckdb-index"]
SPLIT_HAS_PREVIEW_KINDS = ["split-first-rows-from-streaming", "split-first-rows-from-parquet"]
SPLIT_HAS_SEARCH_KINDS = ["split-duckdb-index"]
PARALLEL_STEPS_LISTS = [
    CONFIG_SPLIT_NAMES_KINDS,
    SPLIT_HAS_PREVIEW_KINDS,
]
