# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
from http import HTTPStatus
from typing import Any, Dict, List, Optional

from libcommon.processing_graph import ProcessingGraph, ProcessingGraphSpecification
from libcommon.queue import Queue, Status
from libcommon.simple_cache import upsert_response
from libcommon.state import DatasetState

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
STEP_DATASET_A = "dataset-a"
STEP_CONFIG_B = "config-b"
STEP_SPLIT_C = "split-c"
SIMPLE_PROCESSING_GRAPH_SPECIFICATION: ProcessingGraphSpecification = {
    STEP_DATASET_A: {"input_type": "dataset", "provides_dataset_config_names": True},
    STEP_CONFIG_B: {"input_type": "config", "provides_config_split_names": True, "triggered_by": STEP_DATASET_A},
    STEP_SPLIT_C: {"input_type": "split", "triggered_by": STEP_CONFIG_B},
}

DATASET_GIT_REVISION = "dataset_git_revision"
JOB_RUNNER_VERSION = 1


def get_dataset_state(
    processing_graph: ProcessingGraph,
    dataset: str = DATASET_NAME,
    git_revision: Optional[str] = DATASET_GIT_REVISION,
    error_codes_to_retry: Optional[List[str]] = None,
) -> DatasetState:
    return DatasetState(
        dataset=dataset,
        processing_graph=processing_graph,
        revision=git_revision,
        error_codes_to_retry=error_codes_to_retry,
    )


def assert_equality(value: Any, expected: Any, context: Optional[str] = None) -> None:
    report = {"expected": expected, "got": value}
    if context is not None:
        report["additional"] = context
    assert value == expected, report


def assert_dataset_state(
    dataset_state: DatasetState,
    cache_status: Dict[str, List[str]],
    queue_status: Dict[str, List[str]],
    tasks: List[str],
    config_names: Optional[List[str]] = None,
    split_names_in_first_config: Optional[List[str]] = None,
) -> None:
    if config_names is not None:
        assert_equality(dataset_state.config_names, config_names, context="config_names")
        assert_equality(len(dataset_state.config_states), len(config_names), context="config_states")
        if len(config_names) and split_names_in_first_config is not None:
            assert_equality(
                dataset_state.config_states[0].split_names, split_names_in_first_config, context="split_names"
            )
    computed_cache_status = dataset_state.cache_status.as_response()
    for key, value in cache_status.items():
        assert_equality(computed_cache_status[key], value, key)
    assert_equality(dataset_state.queue_status.as_response(), queue_status, context="queue_status")
    assert_equality(dataset_state.plan.as_response(), tasks, context="tasks")


def put_cache(artifact: str, error_code: Optional[str] = None) -> None:
    parts = artifact.split(",")
    if len(parts) < 2 or len(parts) > 4:
        raise ValueError(f"Unexpected artifact {artifact}: should have at least 2 parts and at most 4")
    step = parts[0]
    dataset = parts[1]
    if len(parts) == 2:
        if not step.startswith("dataset-"):
            raise ValueError(f"Unexpected artifact {artifact}: should start with dataset-")
        content = CONFIG_NAMES_CONTENT
        config = None
        split = None
    elif len(parts) == 3:
        if not step.startswith("config-"):
            raise ValueError(f"Unexpected artifact {artifact}: should start with config-")
        content = SPLIT_NAMES_CONTENT
        config = parts[2]
        split = None
    else:
        if not step.startswith("split-"):
            raise ValueError(f"Unexpected artifact {artifact}: should start with split-")
        content = {}
        config = parts[2]
        split = parts[3]

    if error_code:
        http_status = HTTPStatus.INTERNAL_SERVER_ERROR
        content = {}
    else:
        http_status = HTTPStatus.OK

    upsert_response(
        kind=step,
        dataset=dataset,
        config=config,
        split=split,
        content=content,
        http_status=http_status,
        job_runner_version=JOB_RUNNER_VERSION,
        dataset_git_revision=DATASET_GIT_REVISION,
        error_code=error_code,
    )


def process_next_job(artifact: str) -> None:
    job_type = artifact.split(",")[0]
    job_info = Queue().start_job(job_types_only=[job_type])
    put_cache(artifact)
    Queue().finish_job(job_id=job_info["job_id"], finished_status=Status.SUCCESS)


def compute_all(
    processing_graph: ProcessingGraph,
    dataset: str = DATASET_NAME,
    git_revision: Optional[str] = DATASET_GIT_REVISION,
    error_codes_to_retry: Optional[List[str]] = None,
) -> None:
    dataset_state = get_dataset_state(processing_graph, dataset, git_revision, error_codes_to_retry)
    max_runs = 100
    while dataset_state.should_be_backfilled and max_runs >= 0:
        if max_runs == 0:
            raise ValueError("Too many runs")
        max_runs -= 1
        dataset_state.backfill()
        for task in dataset_state.plan.tasks:
            task_type, sep, artifact = task.id.partition(",")
            if sep is None:
                raise ValueError(f"Unexpected task id {task.id}: should contain a comma")
            if task_type == "CreateJob":
                process_next_job(artifact)
        dataset_state = get_dataset_state(processing_graph, dataset, git_revision, error_codes_to_retry)


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
