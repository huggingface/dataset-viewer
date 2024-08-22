# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

CACHE_COLLECTION_RESPONSES = "cachedResponsesBlue"
CACHE_MONGOENGINE_ALIAS = "cache"
HF_DATASETS_CACHE_APPNAME = "hf_datasets_cache"
PARQUET_METADATA_CACHE_APPNAME = "datasets_server_parquet_metadata"
DESCRIPTIVE_STATISTICS_CACHE_APPNAME = "dataset_viewer_descriptive_statistics"
DUCKDB_INDEX_CACHE_APPNAME = "dataset_viewer_duckdb_index"
DUCKDB_INDEX_JOB_RUNNER_SUBDIRECTORY = "job_runner"
CACHE_METRICS_COLLECTION = "cacheTotalMetric"
TYPE_STATUS_AND_DATASET_STATUS_JOB_COUNTS_COLLECTION = "jobTotalMetric"
WORKER_TYPE_JOB_COUNTS_COLLECTION = "workerTypeJobCounts"
QUEUE_COLLECTION_JOBS = "jobsBlue"
QUEUE_COLLECTION_PAST_JOBS = "pastJobs"
QUEUE_COLLECTION_LOCKS = "locks"
QUEUE_COLLECTION_DATASET_BLOCKAGES = "datasetBlockages"
QUEUE_MONGOENGINE_ALIAS = "queue"
QUEUE_TTL_SECONDS = 600  # 10 minutes
LOCK_TTL_SECONDS_NO_OWNER = 600  # 10 minutes
LOCK_TTL_SECONDS_TO_START_JOB = 600  # 10 minutes
LOCK_TTL_SECONDS_TO_WRITE_ON_GIT_BRANCH = 3600  # 1 hour

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

TAG_NFAA_CONTENT = "not-for-all-audiences"
TAG_NFAA_SYNONYMS = [TAG_NFAA_CONTENT, "nsfw", "porn", "hentai", "inappropriate"]


DEFAULT_MAX_FAILED_RUNS = 3
LARGE_MAX_FAILED_RUNS = 30  # for errors that should not be permanent
MAX_FAILED_RUNS_PER_ERROR_CODE = {
    # default
    "RetryableConfigNamesError": DEFAULT_MAX_FAILED_RUNS,
    "ConnectionError": DEFAULT_MAX_FAILED_RUNS,
    "ExternalServerError": DEFAULT_MAX_FAILED_RUNS,
    "JobManagerCrashedError": DEFAULT_MAX_FAILED_RUNS,
    "StreamingRowsError": DEFAULT_MAX_FAILED_RUNS,
    # "always" retry
    "CreateCommitError": LARGE_MAX_FAILED_RUNS,
    "HfHubError": LARGE_MAX_FAILED_RUNS,
    "LockedDatasetTimeoutError": LARGE_MAX_FAILED_RUNS,
    "PreviousStepStillProcessingError": LARGE_MAX_FAILED_RUNS,
}
ERROR_CODES_TO_RETRY = list(MAX_FAILED_RUNS_PER_ERROR_CODE.keys())

EXTERNAL_DATASET_SCRIPT_PATTERN = "datasets_modules/datasets"

# Arrays are not immutable, we have to take care of not modifying them
# Anyway: in all this file, we allow constant reassignment (no use of Final)
CONFIG_HAS_VIEWER_KIND = "config-size"
CONFIG_INFO_KIND = "config-info"
CONFIG_PARQUET_METADATA_KIND = "config-parquet-metadata"
CONFIG_SPLIT_NAMES_KIND = "config-split-names"
DATASET_CONFIG_NAMES_KIND = "dataset-config-names"
DATASET_INFO_KIND = "dataset-info"
SPLIT_DUCKDB_INDEX_KIND = "split-duckdb-index"
SPLIT_HAS_PREVIEW_KIND = "split-first-rows"
SPLIT_HAS_SEARCH_KIND = "split-duckdb-index"
SPLIT_HAS_STATISTICS_KIND = "split-descriptive-statistics"
ROW_IDX_COLUMN = "__hf_index_id"
HF_FTS_SCORE = "__hf_fts_score"
CROISSANT_MAX_CONFIGS = 100
LOADING_METHODS_MAX_CONFIGS = 100
MAX_NUM_ROWS_PER_PAGE = 100
MAX_COLUMN_NAME_LENGTH = 500

LONG_DURATION_PROMETHEUS_HISTOGRAM_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    25.0,
    50.0,
    75.0,
    100.0,
    250.0,
    500.0,
    750.0,
    1000.0,
    2500.0,
    5000.0,
    float("inf"),
)

YAML_FIELDS_TO_CHECK = ["dataset_info", "configs", "viewer", "language"]
