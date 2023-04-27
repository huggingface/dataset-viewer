# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.processing_graph import ProcessingGraphSpecification

CURRENT_GIT_REVISION = "current_git_revision"

DATASET_NAME = "dataset"

CONFIG_NAME_1 = "config1"
CONFIG_NAME_2 = "config2"
CONFIG_NAMES = [CONFIG_NAME_1, CONFIG_NAME_2]
CONFIG_NAMES_CONTENT = {"config_names": [{"config": config_name} for config_name in CONFIG_NAMES]}

SPLIT_NAME_1 = "split1"
SPLIT_NAME_2 = "split2"
SPLIT_NAMES = [SPLIT_NAME_1, SPLIT_NAME_2]
SPLIT_NAMES_CONTENT = {
    "splits": [{"dataset": DATASET_NAME, "config": CONFIG_NAME_1, "split": split_name} for split_name in SPLIT_NAMES]
}

STEP_VERSION = 1
SIMPLE_PROCESSING_GRAPH_SPECIFICATION: ProcessingGraphSpecification = {
    "dataset-a": {"input_type": "dataset", "provides_dataset_config_names": True, "job_runner_version": STEP_VERSION},
    "config-b": {
        "input_type": "config",
        "provides_config_split_names": True,
        "job_runner_version": STEP_VERSION,
        "triggered_by": "dataset-a",
    },
    "split-c": {"input_type": "split", "job_runner_version": STEP_VERSION, "triggered_by": "config-b"},
}

#     "/config-names": {"input_type": "dataset", "job_runner_version": PROCESSING_STEP_CONFIG_NAMES_VERSION},
#     "/split-names-from-streaming": {
#         "input_type": "config",
#         "requires": "/config-names",
#         "job_runner_version": PROCESSING_STEP_SPLIT_NAMES_FROM_STREAMING_VERSION,
#     },
#     "split-first-rows-from-streaming": {
#         "input_type": "split",
#         "requires": ["/split-names-from-streaming", "/split-names-from-dataset-info"],
#         "required_by_dataset_viewer": True,
#         "job_runner_version": PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_STREAMING_VERSION,
#     },
#     "config-parquet-and-info": {
#         "input_type": "config",
#         "requires": "/config-names",
#         "job_runner_version": PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_VERSION,
#     },
#     "/parquet-and-dataset-info": {
#         "input_type": "dataset",
#         "job_runner_version": PROCESSING_STEP_PARQUET_AND_DATASET_INFO_VERSION,
#     },
#     "config-parquet": {
#         "input_type": "config",
#         "requires": "config-parquet-and-info",
#         "job_runner_version": PROCESSING_STEP_CONFIG_PARQUET_VERSION,
#     },
#     "split-first-rows-from-parquet": {
#         "input_type": "split",
#         "requires": "config-parquet",
#         "job_runner_version": PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_PARQUET_VERSION,
#     },
#     "dataset-parquet": {
#         "input_type": "dataset",
#         "requires": ["config-parquet", "/config-names"],
#         "job_runner_version": PROCESSING_STEP_DATASET_PARQUET_VERSION,
#     },
#     "config-info": {
#         "input_type": "config",
#         "requires": "config-parquet-and-info",
#         "job_runner_version": PROCESSING_STEP_CONFIG_INFO_VERSION,
#     },
#     "dataset-info": {
#         "input_type": "dataset",
#         "requires": ["config-info", "/config-names"],
#         "job_runner_version": PROCESSING_STEP_DATASET_INFO_VERSION,
#     },
#     "/split-names-from-dataset-info": {
#         "input_type": "config",
#         "requires": "config-info",
#         "job_runner_version": PROCESSING_STEP_SPLIT_NAMES_FROM_DATASET_INFO_VERSION,
#     },
#     "config-size": {
#         "input_type": "config",
#         "requires": "config-parquet-and-info",
#         "job_runner_version": PROCESSING_STEP_CONFIG_SIZE_VERSION,
#     },
#     "dataset-size": {
#         "input_type": "dataset",
#         "requires": ["config-size", "/config-names"],
#         "job_runner_version": PROCESSING_STEP_DATASET_SIZE_VERSION,
#     },
#     "dataset-split-names-from-streaming": {
#         "input_type": "dataset",
#         "requires": ["/split-names-from-streaming", "/config-names"],
#         "job_runner_version": PROCESSING_STEP_DATASET_SPLIT_NAMES_FROM_STREAMING_VERSION,
#     },  # to be deprecated
#     "dataset-split-names-from-dataset-info": {
#         "input_type": "dataset",
#         "requires": ["/split-names-from-dataset-info", "/config-names"],
#         "job_runner_version": PROCESSING_STEP_DATASET_SPLIT_NAMES_FROM_DATASET_INFO_VERSION,
#     },  # to be deprecated
#     "dataset-split-names": {
#         "input_type": "dataset",
#         "requires": ["/split-names-from-dataset-info", "/split-names-from-streaming", "/config-names"],
#         "job_runner_version": PROCESSING_STEP_DATASET_SPLIT_NAMES_VERSION,
#     },
#     "dataset-is-valid": {
#         "input_type": "dataset",
#         "requires": [
#             "dataset-split-names",
#             "split-first-rows-from-parquet",
#             "split-first-rows-from-streaming",
#         ],
#         "job_runner_version": PROCESSING_STEP_DATASET_IS_VALID_VERSION,
#     },
#     "split-opt-in-out-urls-scan": {
#         "input_type": "split",
#         "requires": ["split-first-rows-from-streaming"],
#         "job_runner_version": PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION,
#     },
# }
