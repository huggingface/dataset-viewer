# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Dict, List, Optional

from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue
from libcommon.simple_cache import upsert_response
from libcommon.state import DatasetState

DATASET_NAME = "dataset"

REVISION_NAME = "revision"

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


# DATASET_GIT_REVISION = "dataset_git_revision"
# OTHER_DATASET_GIT_REVISION = "other_dataset_git_revision"
JOB_RUNNER_VERSION = 1


def get_dataset_state(
    processing_graph: ProcessingGraph,
    dataset: str = DATASET_NAME,
    revision: str = REVISION_NAME,
    error_codes_to_retry: Optional[List[str]] = None,
) -> DatasetState:
    return DatasetState(
        dataset=dataset,
        revision=revision,
        processing_graph=processing_graph,
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
        assert_equality(computed_cache_status[key], sorted(value), key)
    assert_equality(
        dataset_state.get_queue_status().as_response(),
        {key: sorted(value) for key, value in queue_status.items()},
        context="queue_status",
    )
    assert_equality(dataset_state.plan.as_response(), sorted(tasks), context="tasks")


def put_cache(
    step: str,
    dataset: str,
    revision: str,
    config: Optional[str] = None,
    split: Optional[str] = None,
    error_code: Optional[str] = None,
    use_old_job_runner_version: Optional[bool] = False,
) -> None:
    if not config:
        if not step.startswith("dataset-"):
            raise ValueError("Unexpected artifact: should start with dataset-")
        content = CONFIG_NAMES_CONTENT
        config = None
        split = None
    elif not split:
        if not step.startswith("config-"):
            raise ValueError("Unexpected artifact: should start with config-")
        content = SPLIT_NAMES_CONTENT
        split = None
    else:
        if not step.startswith("split-"):
            raise ValueError("Unexpected artifact: should start with split-")
        content = {}

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
        job_runner_version=JOB_RUNNER_VERSION - 1 if use_old_job_runner_version else JOB_RUNNER_VERSION,
        dataset_git_revision=revision,
        error_code=error_code,
    )


def process_next_job() -> None:
    job_info = Queue().start_job()
    put_cache(
        step=job_info["type"],
        dataset=job_info["params"]["dataset"],
        revision=job_info["params"]["revision"],
        config=job_info["params"]["config"],
        split=job_info["params"]["split"],
    )
    Queue().finish_job(job_id=job_info["job_id"], is_success=True)


def process_all_jobs() -> None:
    runs = 100
    try:
        while runs > 0:
            runs -= 1
            process_next_job()
    except Exception:
        return


def compute_all(
    processing_graph: ProcessingGraph,
    dataset: str = DATASET_NAME,
    revision: str = REVISION_NAME,
    error_codes_to_retry: Optional[List[str]] = None,
) -> None:
    dataset_state = get_dataset_state(processing_graph, dataset, revision, error_codes_to_retry)
    max_runs = 100
    while dataset_state.should_be_backfilled and max_runs >= 0:
        if max_runs == 0:
            raise ValueError("Too many runs")
        max_runs -= 1
        dataset_state.backfill()
        for task in dataset_state.plan.tasks:
            task_type, sep, num = task.id.partition(",")
            if sep is None:
                raise ValueError(f"Unexpected task id {task.id}: should contain a comma")
            if task_type == "CreateJobs":
                process_all_jobs()
        dataset_state = get_dataset_state(processing_graph, dataset, revision, error_codes_to_retry)
