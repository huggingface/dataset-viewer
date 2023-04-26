# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Dict, List, Optional

import pytest

from libcommon.config import ProcessingGraphConfig
from libcommon.constants import (
    PROCESSING_STEP_CONFIG_NAMES_VERSION,
    PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_VERSION,
    PROCESSING_STEP_CONFIG_PARQUET_VERSION,
    PROCESSING_STEP_DATASET_PARQUET_VERSION,
    PROCESSING_STEP_SPLIT_NAMES_FROM_DATASET_INFO_VERSION,
)
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue, Status
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.state import DatasetState

from .utils import (
    CONFIG_NAME_1,
    CONFIG_NAMES,
    CONFIG_NAMES_CONTENT,
    CURRENT_GIT_REVISION,
    DATASET_NAME,
    SPLIT_NAMES,
    SPLIT_NAMES_CONTENT,
)

PROCESSING_GRAPH = ProcessingGraph(processing_graph_specification=ProcessingGraphConfig().specification)


@pytest.fixture(autouse=True)
def queue_mongo_resource_autouse(queue_mongo_resource: QueueMongoResource) -> QueueMongoResource:
    return queue_mongo_resource


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(cache_mongo_resource: CacheMongoResource) -> CacheMongoResource:
    return cache_mongo_resource


def get_dataset_state(
    git_revision: Optional[str] = CURRENT_GIT_REVISION,
    error_codes_to_retry: Optional[List[str]] = None,
) -> DatasetState:
    return DatasetState(
        dataset=DATASET_NAME,
        processing_graph=PROCESSING_GRAPH,
        revision=git_revision,
        error_codes_to_retry=error_codes_to_retry,
    )


def assert_dataset_state(
    config_names: List[str],
    split_names_in_first_config: List[str],
    cache_status: Dict[str, List[str]],
    queue_status: Dict[str, List[str]],
    tasks: List[str],
    git_revision: Optional[str] = CURRENT_GIT_REVISION,
    error_codes_to_retry: Optional[List[str]] = None,
) -> DatasetState:
    dataset_state = get_dataset_state(git_revision=git_revision, error_codes_to_retry=error_codes_to_retry)
    assert dataset_state.config_names == config_names
    assert len(dataset_state.config_states) == len(config_names)
    if len(config_names):
        assert dataset_state.config_states[0].split_names == split_names_in_first_config
    else:
        # this case is just to check the test, not the code
        assert not split_names_in_first_config
    assert dataset_state.cache_status.as_response() == cache_status
    assert dataset_state.queue_status.as_response() == queue_status
    assert dataset_state.plan.as_response() == tasks
    return dataset_state


def test_plan() -> None:
    assert_dataset_state(
        # The config names are not yet known
        config_names=[],
        # The split names are not yet known
        split_names_in_first_config=[],
        # All the dataset-level cache entries are empty
        # No config-level and split-level cache entries is listed, because the config names and splits
        # names are not yet known.
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/config-names,dataset",
                "dataset-info,dataset",
                "dataset-is-valid,dataset",
                "dataset-opt-in-out-urls-count,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [],
        },
        # The queue is empty, so no step is in process.
        queue_status={"in_process": []},
        # The root dataset-level steps, as well as the "fan-in" steps, are ready to be backfilled.
        tasks=[
            "CreateJob[/config-names,dataset]",
            "CreateJob[dataset-info,dataset]",
            "CreateJob[dataset-is-valid,dataset]",
            "CreateJob[dataset-opt-in-out-urls-count,dataset]",
            "CreateJob[dataset-parquet,dataset]",
            "CreateJob[dataset-size,dataset]",
            "CreateJob[dataset-split-names,dataset]",
        ],
    )


def test_plan_job_creation_and_termination() -> None:
    # we launch all the backfill tasks
    dataset_state = get_dataset_state()
    assert dataset_state.plan.as_response() == [
        "CreateJob[/config-names,dataset]",
        "CreateJob[dataset-info,dataset]",
        "CreateJob[dataset-is-valid,dataset]",
        "CreateJob[dataset-opt-in-out-urls-count,dataset]",
        "CreateJob[dataset-parquet,dataset]",
        "CreateJob[dataset-size,dataset]",
        "CreateJob[dataset-split-names,dataset]",
    ]
    dataset_state.backfill()
    assert_dataset_state(
        # The config names are not yet known
        config_names=[],
        # The split names are not yet known
        split_names_in_first_config=[],
        # the cache has not changed
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/config-names,dataset",
                "dataset-info,dataset",
                "dataset-is-valid,dataset",
                "dataset-opt-in-out-urls-count,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [],
        },
        # the jobs have been created and are in process
        queue_status={
            "in_process": [
                "/config-names,dataset",
                "dataset-info,dataset",
                "dataset-is-valid,dataset",
                "dataset-opt-in-out-urls-count,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
            ]
        },
        # thus: no new task
        tasks=[],
    )

    # we simulate the job for "/config-names,dataset" has finished
    job_info = Queue().start_job(job_types_only=["/config-names"])
    upsert_response(
        kind=job_info["type"],
        dataset=job_info["dataset"],
        config=job_info["config"],
        split=job_info["split"],
        content=CONFIG_NAMES_CONTENT,
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_NAMES_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )
    Queue().finish_job(job_id=job_info["job_id"], finished_status=Status.SUCCESS)

    assert_dataset_state(
        # The config names are now known
        config_names=CONFIG_NAMES,
        # The split names are not yet known
        split_names_in_first_config=[],
        # The "/config-names" step is up-to-date
        # Config-level artifacts are empty and ready to be filled (even if some of their parents are still missing)
        # The split-level artifacts are still missing, because the splits names are not yet known, for any config.
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/split-names-from-dataset-info,dataset,config1",
                "/split-names-from-dataset-info,dataset,config2",
                "/split-names-from-streaming,dataset,config1",
                "/split-names-from-streaming,dataset,config2",
                "config-info,dataset,config1",
                "config-info,dataset,config2",
                "config-opt-in-out-urls-count,dataset,config1",
                "config-opt-in-out-urls-count,dataset,config2",
                "config-parquet,dataset,config1",
                "config-parquet,dataset,config2",
                "config-parquet-and-info,dataset,config1",
                "config-parquet-and-info,dataset,config2",
                "config-size,dataset,config1",
                "config-size,dataset,config2",
                "dataset-info,dataset",
                "dataset-is-valid,dataset",
                "dataset-opt-in-out-urls-count,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": ["/config-names,dataset"],
        },
        # the job "/config-names,dataset" is no more in process
        queue_status={
            "in_process": [
                "dataset-info,dataset",
                "dataset-is-valid,dataset",
                "dataset-opt-in-out-urls-count,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
            ]
        },
        tasks=[
            "CreateJob[/split-names-from-dataset-info,dataset,config1]",
            "CreateJob[/split-names-from-dataset-info,dataset,config2]",
            "CreateJob[/split-names-from-streaming,dataset,config1]",
            "CreateJob[/split-names-from-streaming,dataset,config2]",
            "CreateJob[config-info,dataset,config1]",
            "CreateJob[config-info,dataset,config2]",
            "CreateJob[config-opt-in-out-urls-count,dataset,config1]",
            "CreateJob[config-opt-in-out-urls-count,dataset,config2]",
            "CreateJob[config-parquet,dataset,config1]",
            "CreateJob[config-parquet,dataset,config2]",
            "CreateJob[config-parquet-and-info,dataset,config1]",
            "CreateJob[config-parquet-and-info,dataset,config2]",
            "CreateJob[config-size,dataset,config1]",
            "CreateJob[config-size,dataset,config2]",
        ],
    )


def test_plan_retry_error() -> None:
    ERROR_CODE_TO_RETRY = "ERROR_CODE_TO_RETRY"
    # Set the "/config-names,dataset" artifact in cache
    upsert_response(
        kind="/config-names",
        dataset=DATASET_NAME,
        config=None,
        split=None,
        content=CONFIG_NAMES_CONTENT,
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
        job_runner_version=PROCESSING_STEP_CONFIG_NAMES_VERSION,
        error_code=ERROR_CODE_TO_RETRY,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )

    assert_dataset_state(
        error_codes_to_retry=[ERROR_CODE_TO_RETRY],
        # The config names are known
        config_names=CONFIG_NAMES,
        # The split names are not yet known
        split_names_in_first_config=[],
        # "/config-names,dataset" is in the cache, but it's not categorized in up to date,
        # but in "cache_is_error_to_retry" due to the error code
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/split-names-from-dataset-info,dataset,config1",
                "/split-names-from-dataset-info,dataset,config2",
                "/split-names-from-streaming,dataset,config1",
                "/split-names-from-streaming,dataset,config2",
                "config-info,dataset,config1",
                "config-info,dataset,config2",
                "config-parquet,dataset,config1",
                "config-parquet,dataset,config2",
                "config-parquet-and-info,dataset,config1",
                "config-parquet-and-info,dataset,config2",
                "config-size,dataset,config1",
                "config-size,dataset,config2",
                "dataset-info,dataset",
                "dataset-is-valid,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
            ],
            "cache_is_error_to_retry": ["/config-names,dataset"],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [],
        },
        queue_status={"in_process": []},
        # The "/config-names,dataset" artifact will be retried
        tasks=[
            "CreateJob[/config-names,dataset]",
            "CreateJob[/split-names-from-dataset-info,dataset,config1]",
            "CreateJob[/split-names-from-dataset-info,dataset,config2]",
            "CreateJob[/split-names-from-streaming,dataset,config1]",
            "CreateJob[/split-names-from-streaming,dataset,config2]",
            "CreateJob[config-info,dataset,config1]",
            "CreateJob[config-info,dataset,config2]",
            "CreateJob[config-opt-in-out-urls-count,dataset,config1]",
            "CreateJob[config-opt-in-out-urls-count,dataset,config2]",
            "CreateJob[config-parquet,dataset,config1]",
            "CreateJob[config-parquet,dataset,config2]",
            "CreateJob[config-parquet-and-info,dataset,config1]",
            "CreateJob[config-parquet-and-info,dataset,config2]",
            "CreateJob[config-size,dataset,config1]",
            "CreateJob[config-size,dataset,config2]",
            "CreateJob[dataset-info,dataset]",
            "CreateJob[dataset-is-valid,dataset]",
            "CreateJob[dataset-opt-in-out-urls-count,dataset]",
            "CreateJob[dataset-parquet,dataset]",
            "CreateJob[dataset-size,dataset]",
            "CreateJob[dataset-split-names,dataset]",
        ],
    )


def test_plan_incoherent_state() -> None:
    # Set the "/config-names,dataset" artifact in cache
    upsert_response(
        kind="/config-names",
        dataset=DATASET_NAME,
        config=None,
        split=None,
        content=CONFIG_NAMES_CONTENT,
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_NAMES_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )
    # Set the "/split-names-from-dataset-info,dataset,config1" artifact in cache
    # -> It's not a coherent state for the cache: the ancestors artifacts are missing:
    # "config-parquet-and-info,dataset" and "config-info,dataset,config1"
    upsert_response(
        kind="/split-names-from-dataset-info",
        dataset=DATASET_NAME,
        config=CONFIG_NAME_1,
        split=None,
        content=SPLIT_NAMES_CONTENT,
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_SPLIT_NAMES_FROM_DATASET_INFO_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )

    assert_dataset_state(
        # The config names are known
        config_names=CONFIG_NAMES,
        # The split names are known
        split_names_in_first_config=SPLIT_NAMES,
        # The split level artifacts for config1 are ready to be backfilled
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/split-names-from-dataset-info,dataset,config2",
                "/split-names-from-streaming,dataset,config1",
                "/split-names-from-streaming,dataset,config2",
                "config-info,dataset,config1",
                "config-info,dataset,config2",
                "config-opt-in-out-urls-count,dataset,config1",
                "config-opt-in-out-urls-count,dataset,config2",
                "config-parquet,dataset,config1",
                "config-parquet,dataset,config2",
                "config-parquet-and-info,dataset,config1",
                "config-parquet-and-info,dataset,config2",
                "config-size,dataset,config1",
                "config-size,dataset,config2",
                "dataset-info,dataset",
                "dataset-is-valid,dataset",
                "dataset-opt-in-out-urls-count,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "split-first-rows-from-parquet,dataset,config1,split1",
                "split-first-rows-from-parquet,dataset,config1,split2",
                "split-first-rows-from-streaming,dataset,config1,split1",
                "split-first-rows-from-streaming,dataset,config1,split2",
                "split-opt-in-out-urls-count,dataset,config1,split1",
                "split-opt-in-out-urls-count,dataset,config1,split2",
                "split-opt-in-out-urls-scan,dataset,config1,split1",
                "split-opt-in-out-urls-scan,dataset,config1,split2",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": ["/config-names,dataset", "/split-names-from-dataset-info,dataset,config1"],
        },
        queue_status={"in_process": []},
        tasks=[
            "CreateJob[/split-names-from-dataset-info,dataset,config2]",
            "CreateJob[/split-names-from-streaming,dataset,config1]",
            "CreateJob[/split-names-from-streaming,dataset,config2]",
            "CreateJob[config-info,dataset,config1]",
            "CreateJob[config-info,dataset,config2]",
            "CreateJob[config-opt-in-out-urls-count,dataset,config1]",
            "CreateJob[config-opt-in-out-urls-count,dataset,config2]",
            "CreateJob[config-parquet,dataset,config1]",
            "CreateJob[config-parquet,dataset,config2]",
            "CreateJob[config-parquet-and-info,dataset,config1]",
            "CreateJob[config-parquet-and-info,dataset,config2]",
            "CreateJob[config-size,dataset,config1]",
            "CreateJob[config-size,dataset,config2]",
            "CreateJob[dataset-info,dataset]",
            "CreateJob[dataset-is-valid,dataset]",
            "CreateJob[dataset-opt-in-out-urls-count,dataset]",
            "CreateJob[dataset-parquet,dataset]",
            "CreateJob[dataset-size,dataset]",
            "CreateJob[dataset-split-names,dataset]",
            "CreateJob[split-first-rows-from-parquet,dataset,config1,split1]",
            "CreateJob[split-first-rows-from-parquet,dataset,config1,split2]",
            "CreateJob[split-first-rows-from-streaming,dataset,config1,split1]",
            "CreateJob[split-first-rows-from-streaming,dataset,config1,split2]",
            "CreateJob[split-opt-in-out-urls-count,dataset,config1,split1]",
            "CreateJob[split-opt-in-out-urls-count,dataset,config1,split2]",
            "CreateJob[split-opt-in-out-urls-scan,dataset,config1,split1]",
            "CreateJob[split-opt-in-out-urls-scan,dataset,config1,split2]",
        ],
    )


def test_plan_updated_at() -> None:
    # Set the "/config-names,dataset" artifact in cache
    upsert_response(
        kind="/config-names",
        dataset=DATASET_NAME,
        config=None,
        split=None,
        content=CONFIG_NAMES_CONTENT,
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_NAMES_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )
    # Set the "config-parquet-and-info,dataset,config1" artifact in cache
    upsert_response(
        kind="config-parquet-and-info",
        dataset=DATASET_NAME,
        config=CONFIG_NAME_1,
        split=None,
        content=CONFIG_NAMES_CONTENT,  # <- not important
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )
    # Now: refresh again the "/config-names,dataset" artifact in cache
    upsert_response(
        kind="/config-names",
        dataset=DATASET_NAME,
        config=None,
        split=None,
        content=CONFIG_NAMES_CONTENT,
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_NAMES_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )

    assert_dataset_state(
        # The config names are known
        config_names=CONFIG_NAMES,
        # The split names are not yet known
        split_names_in_first_config=[],
        # config-parquet-and-info,dataset,config1 is marked as outdated by parent,
        # Only "/config-names,dataset" is marked as up to date
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": ["config-parquet-and-info,dataset,config1"],
            "cache_is_empty": [
                "/split-names-from-dataset-info,dataset,config1",
                "/split-names-from-dataset-info,dataset,config2",
                "/split-names-from-streaming,dataset,config1",
                "/split-names-from-streaming,dataset,config2",
                "config-info,dataset,config1",
                "config-info,dataset,config2",
                "config-opt-in-out-urls-count,dataset,config1",
                "config-opt-in-out-urls-count,dataset,config2",
                "config-parquet,dataset,config1",
                "config-parquet,dataset,config2",
                "config-parquet-and-info,dataset,config2",
                "config-size,dataset,config1",
                "config-size,dataset,config2",
                "dataset-info,dataset",
                "dataset-is-valid,dataset",
                "dataset-opt-in-out-urls-count,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": ["/config-names,dataset"],
        },
        queue_status={"in_process": []},
        # config-parquet-and-info,dataset,config1 will be refreshed
        tasks=[
            "CreateJob[/split-names-from-dataset-info,dataset,config1]",
            "CreateJob[/split-names-from-dataset-info,dataset,config2]",
            "CreateJob[/split-names-from-streaming,dataset,config1]",
            "CreateJob[/split-names-from-streaming,dataset,config2]",
            "CreateJob[config-info,dataset,config1]",
            "CreateJob[config-info,dataset,config2]",
            "CreateJob[config-opt-in-out-urls-count,dataset,config1]",
            "CreateJob[config-opt-in-out-urls-count,dataset,config2]",
            "CreateJob[config-parquet,dataset,config1]",
            "CreateJob[config-parquet,dataset,config2]",
            "CreateJob[config-parquet-and-info,dataset,config1]",
            "CreateJob[config-parquet-and-info,dataset,config2]",
            "CreateJob[config-size,dataset,config1]",
            "CreateJob[config-size,dataset,config2]",
            "CreateJob[dataset-info,dataset]",
            "CreateJob[dataset-is-valid,dataset]",
            "CreateJob[dataset-opt-in-out-urls-count,dataset]",
            "CreateJob[dataset-parquet,dataset]",
            "CreateJob[dataset-size,dataset]",
            "CreateJob[dataset-split-names,dataset]",
        ],
    )


def test_plan_job_runner_version() -> None:
    # Set the "/config-names,dataset" artifact in cache
    upsert_response(
        kind="/config-names",
        dataset=DATASET_NAME,
        config=None,
        split=None,
        content=CONFIG_NAMES_CONTENT,
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_NAMES_VERSION - 1,  # <- old version
        dataset_git_revision=CURRENT_GIT_REVISION,
    )
    assert_dataset_state(
        # The config names are known
        config_names=CONFIG_NAMES,
        # The split names are not known
        split_names_in_first_config=[],
        # /config-names is in the category: "is_job_runner_obsolete"
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/split-names-from-dataset-info,dataset,config1",
                "/split-names-from-dataset-info,dataset,config2",
                "/split-names-from-streaming,dataset,config1",
                "/split-names-from-streaming,dataset,config2",
                "config-info,dataset,config1",
                "config-info,dataset,config2",
                "config-opt-in-out-urls-count,dataset,config1",
                "config-opt-in-out-urls-count,dataset,config2",
                "config-parquet,dataset,config1",
                "config-parquet,dataset,config2",
                "config-parquet-and-info,dataset,config1",
                "config-parquet-and-info,dataset,config2",
                "config-size,dataset,config1",
                "config-size,dataset,config2",
                "dataset-info,dataset",
                "dataset-is-valid,dataset",
                "dataset-opt-in-out-urls-count,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": ["/config-names,dataset"],
            "up_to_date": [],
        },
        queue_status={"in_process": []},
        # "/config-names,dataset" will be refreshed because its job runner has been upgraded
        tasks=[
            "CreateJob[/config-names,dataset]",
            "CreateJob[/split-names-from-dataset-info,dataset,config1]",
            "CreateJob[/split-names-from-dataset-info,dataset,config2]",
            "CreateJob[/split-names-from-streaming,dataset,config1]",
            "CreateJob[/split-names-from-streaming,dataset,config2]",
            "CreateJob[config-info,dataset,config1]",
            "CreateJob[config-info,dataset,config2]",
            "CreateJob[config-opt-in-out-urls-count,dataset,config1]",
            "CreateJob[config-opt-in-out-urls-count,dataset,config2]",
            "CreateJob[config-parquet,dataset,config1]",
            "CreateJob[config-parquet,dataset,config2]",
            "CreateJob[config-parquet-and-info,dataset,config1]",
            "CreateJob[config-parquet-and-info,dataset,config2]",
            "CreateJob[config-size,dataset,config1]",
            "CreateJob[config-size,dataset,config2]",
            "CreateJob[dataset-info,dataset]",
            "CreateJob[dataset-is-valid,dataset]",
            "CreateJob[dataset-opt-in-out-urls-count,dataset]",
            "CreateJob[dataset-parquet,dataset]",
            "CreateJob[dataset-size,dataset]",
            "CreateJob[dataset-split-names,dataset]",
        ],
    )


@pytest.mark.parametrize(
    "dataset_git_revision,cached_dataset_get_revision,expect_refresh",
    [
        (None, None, False),
        ("a", "a", False),
        (None, "b", True),
        ("a", None, True),
        ("a", "b", True),
    ],
)
def test_plan_git_revision(
    dataset_git_revision: Optional[str], cached_dataset_get_revision: Optional[str], expect_refresh: bool
) -> None:
    # Set the "/config-names,dataset" artifact in cache
    upsert_response(
        kind="/config-names",
        dataset=DATASET_NAME,
        config=None,
        split=None,
        content=CONFIG_NAMES_CONTENT,
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_NAMES_VERSION,
        dataset_git_revision=cached_dataset_get_revision,
    )

    if expect_refresh:
        # if the git revision is different from the current dataset git revision, the artifact will be refreshed
        assert_dataset_state(
            git_revision=dataset_git_revision,
            # The config names are known
            config_names=CONFIG_NAMES,
            # The split names are not known
            split_names_in_first_config=[],
            cache_status={
                "cache_has_different_git_revision": ["/config-names,dataset"],
                "cache_is_outdated_by_parent": [],
                "cache_is_empty": [
                    "/split-names-from-dataset-info,dataset,config1",
                    "/split-names-from-dataset-info,dataset,config2",
                    "/split-names-from-streaming,dataset,config1",
                    "/split-names-from-streaming,dataset,config2",
                    "config-info,dataset,config1",
                    "config-info,dataset,config2",
                    "config-opt-in-out-urls-count,dataset,config1",
                    "config-opt-in-out-urls-count,dataset,config2",
                    "config-parquet,dataset,config1",
                    "config-parquet,dataset,config2",
                    "config-parquet-and-info,dataset,config1",
                    "config-parquet-and-info,dataset,config2",
                    "config-size,dataset,config1",
                    "config-size,dataset,config2",
                    "dataset-info,dataset",
                    "dataset-is-valid,dataset",
                    "dataset-opt-in-out-urls-count,dataset",
                    "dataset-parquet,dataset",
                    "dataset-size,dataset",
                    "dataset-split-names,dataset",
                ],
                "cache_is_error_to_retry": [],
                "cache_is_job_runner_obsolete": [],
                "up_to_date": [],
            },
            queue_status={"in_process": []},
            tasks=[
                "CreateJob[/config-names,dataset]",
                "CreateJob[/split-names-from-dataset-info,dataset,config1]",
                "CreateJob[/split-names-from-dataset-info,dataset,config2]",
                "CreateJob[/split-names-from-streaming,dataset,config1]",
                "CreateJob[/split-names-from-streaming,dataset,config2]",
                "CreateJob[config-info,dataset,config1]",
                "CreateJob[config-info,dataset,config2]",
                "CreateJob[config-opt-in-out-urls-count,dataset,config1]",
                "CreateJob[config-opt-in-out-urls-count,dataset,config2]",
                "CreateJob[config-parquet,dataset,config1]",
                "CreateJob[config-parquet,dataset,config2]",
                "CreateJob[config-parquet-and-info,dataset,config1]",
                "CreateJob[config-parquet-and-info,dataset,config2]",
                "CreateJob[config-size,dataset,config1]",
                "CreateJob[config-size,dataset,config2]",
                "CreateJob[dataset-info,dataset]",
                "CreateJob[dataset-is-valid,dataset]",
                "CreateJob[dataset-opt-in-out-urls-count,dataset]",
                "CreateJob[dataset-parquet,dataset]",
                "CreateJob[dataset-size,dataset]",
                "CreateJob[dataset-split-names,dataset]",
            ],
        )
    else:
        assert_dataset_state(
            git_revision=dataset_git_revision,
            # The config names are known
            config_names=CONFIG_NAMES,
            # The split names are not known
            split_names_in_first_config=[],
            cache_status={
                "cache_has_different_git_revision": [],
                "cache_is_outdated_by_parent": [],
                "cache_is_empty": [
                    "/split-names-from-dataset-info,dataset,config1",
                    "/split-names-from-dataset-info,dataset,config2",
                    "/split-names-from-streaming,dataset,config1",
                    "/split-names-from-streaming,dataset,config2",
                    "config-info,dataset,config1",
                    "config-info,dataset,config2",
                    "config-opt-in-out-urls-count,dataset,config1",
                    "config-opt-in-out-urls-count,dataset,config2",
                    "config-parquet,dataset,config1",
                    "config-parquet,dataset,config2",
                    "config-parquet-and-info,dataset,config1",
                    "config-parquet-and-info,dataset,config2",
                    "config-size,dataset,config1",
                    "config-size,dataset,config2",
                    "dataset-info,dataset",
                    "dataset-is-valid,dataset",
                    "dataset-opt-in-out-urls-count,dataset",
                    "dataset-parquet,dataset",
                    "dataset-size,dataset",
                    "dataset-split-names,dataset",
                ],
                "cache_is_error_to_retry": [],
                "cache_is_job_runner_obsolete": [],
                "up_to_date": ["/config-names,dataset"],
            },
            queue_status={"in_process": []},
            tasks=[
                "CreateJob[/split-names-from-dataset-info,dataset,config1]",
                "CreateJob[/split-names-from-dataset-info,dataset,config2]",
                "CreateJob[/split-names-from-streaming,dataset,config1]",
                "CreateJob[/split-names-from-streaming,dataset,config2]",
                "CreateJob[config-info,dataset,config1]",
                "CreateJob[config-info,dataset,config2]",
                "CreateJob[config-opt-in-out-urls-count,dataset,config1]",
                "CreateJob[config-opt-in-out-urls-count,dataset,config2]",
                "CreateJob[config-parquet,dataset,config1]",
                "CreateJob[config-parquet,dataset,config2]",
                "CreateJob[config-parquet-and-info,dataset,config1]",
                "CreateJob[config-parquet-and-info,dataset,config2]",
                "CreateJob[config-size,dataset,config1]",
                "CreateJob[config-size,dataset,config2]",
                "CreateJob[dataset-info,dataset]",
                "CreateJob[dataset-is-valid,dataset]",
                "CreateJob[dataset-opt-in-out-urls-count,dataset]",
                "CreateJob[dataset-parquet,dataset]",
                "CreateJob[dataset-size,dataset]",
                "CreateJob[dataset-split-names,dataset]",
            ],
        )


def test_plan_update_fan_in_parent() -> None:
    # Set the "/config-names,dataset" artifact in cache
    upsert_response(
        kind="/config-names",
        dataset=DATASET_NAME,
        config=None,
        split=None,
        content=CONFIG_NAMES_CONTENT,
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_NAMES_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )
    # Set the "dataset-parquet,dataset" artifact in cache
    upsert_response(
        kind="dataset-parquet",
        dataset=DATASET_NAME,
        config=None,
        split=None,
        content=CONFIG_NAMES_CONTENT,  # <- not important
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_DATASET_PARQUET_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )
    # Set the "config-parquet-and-info,dataset,config1" artifact in cache
    upsert_response(
        kind="config-parquet-and-info",
        dataset=DATASET_NAME,
        config=CONFIG_NAME_1,
        split=None,
        content=CONFIG_NAMES_CONTENT,  # <- not important
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )
    # Set the "config-parquet,dataset,config1" artifact in cache
    upsert_response(
        kind="config-parquet",
        dataset=DATASET_NAME,
        config=CONFIG_NAME_1,
        split=None,
        content=CONFIG_NAMES_CONTENT,  # <- not important
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_PARQUET_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )
    assert_dataset_state(
        # The config names are known
        config_names=CONFIG_NAMES,
        # The split names are not known
        split_names_in_first_config=[],
        # dataset-parquet,dataset is in the category: "cache_is_outdated_by_parent"
        # because one of the "config-parquet" artifacts is more recent
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [
                "dataset-parquet,dataset",
            ],
            "cache_is_empty": [
                "/split-names-from-dataset-info,dataset,config1",
                "/split-names-from-dataset-info,dataset,config2",
                "/split-names-from-streaming,dataset,config1",
                "/split-names-from-streaming,dataset,config2",
                "config-info,dataset,config1",
                "config-info,dataset,config2",
                "config-opt-in-out-urls-count,dataset,config1",
                "config-opt-in-out-urls-count,dataset,config2",
                "config-parquet,dataset,config2",
                "config-parquet-and-info,dataset,config2",
                "config-size,dataset,config1",
                "config-size,dataset,config2",
                "dataset-info,dataset",
                "dataset-is-valid,dataset",
                "dataset-opt-in-out-urls-count,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [
                "/config-names,dataset",
                "config-parquet,dataset,config1",
                "config-parquet-and-info,dataset,config1",
            ],
        },
        queue_status={"in_process": []},
        # dataset-parquet,dataset will be refreshed
        tasks=[
            "CreateJob[/split-names-from-dataset-info,dataset,config1]",
            "CreateJob[/split-names-from-dataset-info,dataset,config2]",
            "CreateJob[/split-names-from-streaming,dataset,config1]",
            "CreateJob[/split-names-from-streaming,dataset,config2]",
            "CreateJob[config-info,dataset,config1]",
            "CreateJob[config-info,dataset,config2]",
            "CreateJob[config-opt-in-out-urls-count,dataset,config1]",
            "CreateJob[config-opt-in-out-urls-count,dataset,config2]",
            "CreateJob[config-parquet,dataset,config2]",
            "CreateJob[config-parquet-and-info,dataset,config2]",
            "CreateJob[config-size,dataset,config1]",
            "CreateJob[config-size,dataset,config2]",
            "CreateJob[dataset-info,dataset]",
            "CreateJob[dataset-is-valid,dataset]",
            "CreateJob[dataset-opt-in-out-urls-count,dataset]",
            "CreateJob[dataset-parquet,dataset]",
            "CreateJob[dataset-size,dataset]",
            "CreateJob[dataset-split-names,dataset]",
        ],
    )
