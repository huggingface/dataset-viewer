# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Dict, List, Mapping, Optional, TypedDict
from unittest.mock import patch

import pytest

from libcommon.config import ProcessingGraphConfig
from libcommon.constants import (
    PROCESSING_STEP_CONFIG_INFO_VERSION,
    PROCESSING_STEP_CONFIG_NAMES_VERSION,
    PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_VERSION,
    PROCESSING_STEP_CONFIG_PARQUET_VERSION,
    PROCESSING_STEP_SPLIT_NAMES_FROM_DATASET_INFO_VERSION,
    PROCESSING_STEP_SPLIT_NAMES_FROM_STREAMING_VERSION,
)
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue, Status
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.state import (
    HARD_CODED_CONFIG_NAMES_CACHE_KIND,
    HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND,
    HARD_CODED_SPLIT_NAMES_FROM_STREAMING_CACHE_KIND,
    ArtifactState,
    CacheState,
    ConfigState,
    DatasetState,
    JobState,
    SplitState,
    fetch_config_names,
    fetch_split_names,
)


@pytest.fixture(autouse=True)
def queue_mongo_resource_autouse(queue_mongo_resource: QueueMongoResource) -> QueueMongoResource:
    return queue_mongo_resource


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(cache_mongo_resource: CacheMongoResource) -> CacheMongoResource:
    return cache_mongo_resource


DATASET_NAME = "dataset"
CONFIG_NAMES_OK = ["config1", "config2"]
CONFIG_NAMES_CONTENT_OK = {"config_names": [{"config": config_name} for config_name in CONFIG_NAMES_OK]}
CONTENT_ERROR = {"error": "error"}


@pytest.mark.parametrize(
    "content,http_status,expected_config_names",
    [
        (CONFIG_NAMES_CONTENT_OK, HTTPStatus.OK, CONFIG_NAMES_OK),
        (CONTENT_ERROR, HTTPStatus.INTERNAL_SERVER_ERROR, None),
        (None, HTTPStatus.OK, None),
    ],
)
def test_fetch_config_names(
    content: Optional[Mapping[str, Any]], http_status: HTTPStatus, expected_config_names: Optional[List[str]]
) -> None:
    raises = expected_config_names is None
    if content:
        upsert_response(
            kind=HARD_CODED_CONFIG_NAMES_CACHE_KIND,
            dataset=DATASET_NAME,
            config=None,
            split=None,
            content=content,
            http_status=http_status,
        )

    if raises:
        with pytest.raises(Exception):
            fetch_config_names(dataset=DATASET_NAME)
    else:
        config_names = fetch_config_names(dataset=DATASET_NAME)
        assert config_names == expected_config_names


class ResponseSpec(TypedDict):
    content: Mapping[str, Any]
    http_status: HTTPStatus


CONFIG_NAME_1 = "config1"
SPLIT_NAMES_OK = ["split1", "split2"]


def get_SPLIT_NAMES_CONTENT_OK(dataset: str, config: str, splits: List[str]) -> Any:
    return {"split_names": [{"dataset": dataset, "config": config, "split": split_name} for split_name in splits]}


SPLIT_NAMES_RESPONSE_OK = ResponseSpec(
    content=get_SPLIT_NAMES_CONTENT_OK(dataset=DATASET_NAME, config=CONFIG_NAME_1, splits=SPLIT_NAMES_OK),
    http_status=HTTPStatus.OK,
)
SPLIT_NAMES_RESPONSE_ERROR = ResponseSpec(content={"error": "error"}, http_status=HTTPStatus.INTERNAL_SERVER_ERROR)


@pytest.mark.parametrize(
    "response_spec_by_kind,expected_split_names",
    [
        ({HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND: SPLIT_NAMES_RESPONSE_OK}, SPLIT_NAMES_OK),
        ({HARD_CODED_SPLIT_NAMES_FROM_STREAMING_CACHE_KIND: SPLIT_NAMES_RESPONSE_OK}, SPLIT_NAMES_OK),
        (
            {
                HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND: SPLIT_NAMES_RESPONSE_ERROR,
                HARD_CODED_SPLIT_NAMES_FROM_STREAMING_CACHE_KIND: SPLIT_NAMES_RESPONSE_OK,
            },
            SPLIT_NAMES_OK,
        ),
        ({HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND: SPLIT_NAMES_RESPONSE_ERROR}, None),
        ({}, None),
    ],
)
def test_fetch_split_names(
    response_spec_by_kind: Mapping[str, Mapping[str, Any]],
    expected_split_names: Optional[List[str]],
) -> None:
    raises = expected_split_names is None
    for kind, response_spec in response_spec_by_kind.items():
        upsert_response(
            kind=kind,
            dataset=DATASET_NAME,
            config=CONFIG_NAME_1,
            split=None,
            content=response_spec["content"],
            http_status=response_spec["http_status"],
        )

    if raises:
        with pytest.raises(Exception):
            fetch_split_names(dataset=DATASET_NAME, config=CONFIG_NAME_1)
    else:
        split_names = fetch_split_names(dataset=DATASET_NAME, config=CONFIG_NAME_1)
        assert split_names == expected_split_names


SPLIT_NAME = "split"
JOB_TYPE = "job_type"


@pytest.mark.parametrize(
    "dataset,config,split,job_type",
    [
        (DATASET_NAME, None, None, JOB_TYPE),
        (DATASET_NAME, CONFIG_NAME_1, None, JOB_TYPE),
        (DATASET_NAME, CONFIG_NAME_1, SPLIT_NAME, JOB_TYPE),
    ],
)
def test_job_state_is_in_process(dataset: str, config: Optional[str], split: Optional[str], job_type: str) -> None:
    queue = Queue()
    queue.upsert_job(job_type=job_type, dataset=dataset, config=config, split=split)
    assert JobState(dataset=dataset, config=config, split=split, job_type=job_type).is_in_process
    job_info = queue.start_job()
    assert JobState(dataset=dataset, config=config, split=split, job_type=job_type).is_in_process
    queue.finish_job(job_id=job_info["job_id"], finished_status=Status.SUCCESS)
    assert not JobState(dataset=dataset, config=config, split=split, job_type=job_type).is_in_process


@pytest.mark.parametrize(
    "dataset,config,split,job_type",
    [
        (DATASET_NAME, None, None, JOB_TYPE),
        (DATASET_NAME, CONFIG_NAME_1, None, JOB_TYPE),
        (DATASET_NAME, CONFIG_NAME_1, SPLIT_NAME, JOB_TYPE),
    ],
)
def test_job_state_as_dict(dataset: str, config: Optional[str], split: Optional[str], job_type: str) -> None:
    queue = Queue()
    queue.upsert_job(job_type=job_type, dataset=dataset, config=config, split=split)
    assert JobState(dataset=dataset, config=config, split=split, job_type=job_type).as_dict() == {
        "is_in_process": True,
    }


CACHE_KIND = "cache_kind"


@pytest.mark.parametrize(
    "dataset,config,split,cache_kind",
    [
        (DATASET_NAME, None, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME_1, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME_1, SPLIT_NAME, CACHE_KIND),
    ],
)
def test_cache_state_exists(dataset: str, config: Optional[str], split: Optional[str], cache_kind: str) -> None:
    assert not CacheState(dataset=dataset, config=config, split=split, cache_kind=cache_kind).exists
    upsert_response(
        kind=cache_kind, dataset=dataset, config=config, split=split, content={}, http_status=HTTPStatus.OK
    )
    assert CacheState(dataset=dataset, config=config, split=split, cache_kind=cache_kind).exists


@pytest.mark.parametrize(
    "dataset,config,split,cache_kind",
    [
        (DATASET_NAME, None, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME_1, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME_1, SPLIT_NAME, CACHE_KIND),
    ],
)
def test_cache_state_is_success(dataset: str, config: Optional[str], split: Optional[str], cache_kind: str) -> None:
    upsert_response(
        kind=cache_kind, dataset=dataset, config=config, split=split, content={}, http_status=HTTPStatus.OK
    )
    assert CacheState(dataset=dataset, config=config, split=split, cache_kind=cache_kind).is_success
    upsert_response(
        kind=cache_kind,
        dataset=dataset,
        config=config,
        split=split,
        content={},
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    )
    assert not CacheState(dataset=dataset, config=config, split=split, cache_kind=cache_kind).is_success


@pytest.mark.parametrize(
    "dataset,config,split,cache_kind",
    [
        (DATASET_NAME, None, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME_1, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME_1, SPLIT_NAME, CACHE_KIND),
    ],
)
def test_cache_state_as_dict(dataset: str, config: Optional[str], split: Optional[str], cache_kind: str) -> None:
    assert CacheState(dataset=dataset, config=config, split=split, cache_kind=cache_kind).as_dict() == {
        "exists": False,
        "is_success": False,
    }
    upsert_response(
        kind=cache_kind,
        dataset=dataset,
        config=config,
        split=split,
        content={"some": "content"},
        http_status=HTTPStatus.OK,
    )
    assert CacheState(dataset=dataset, config=config, split=split, cache_kind=cache_kind).as_dict() == {
        "exists": True,
        "is_success": True,
    }


PROCESSING_GRAPH = ProcessingGraph(processing_graph_specification=ProcessingGraphConfig().specification)


def test_artifact_state() -> None:
    dataset = DATASET_NAME
    config = None
    split = None
    step = PROCESSING_GRAPH.get_step(name="/config-names")
    artifact_state = ArtifactState(dataset=dataset, config=config, split=split, step=step)
    assert artifact_state.as_dict() == {
        "id": f"/config-names,{dataset}",
        "job_state": {"is_in_process": False},
        "cache_state": {"exists": False, "is_success": False},
    }
    assert not artifact_state.cache_state.exists
    assert not artifact_state.job_state.is_in_process


def get_SPLIT_STATE_DICT(dataset: str, config: str, split: str) -> Any:
    return {
        "split": split,
        "artifact_states": [
            {
                "id": f"split-first-rows-from-streaming,{dataset},{config},{split}",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
            },
            {
                "id": f"split-first-rows-from-parquet,{dataset},{config},{split}",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
            },
        ],
    }


SPLIT1_NAME = "split1"


def test_split_state_as_dict() -> None:
    dataset = DATASET_NAME
    config = CONFIG_NAME_1
    split = SPLIT1_NAME
    processing_graph = PROCESSING_GRAPH
    assert SplitState(
        dataset=dataset, config=config, split=split, processing_graph=processing_graph
    ).as_dict() == get_SPLIT_STATE_DICT(dataset=dataset, config=config, split=split)


SPLIT2_NAME = "split2"


def get_CONFIG_STATE_DICT(dataset: str, config: str, split_states: List[Any], cache_exists: bool) -> Any:
    return {
        "config": config,
        "split_states": split_states,
        "artifact_states": [
            {
                "id": f"/split-names-from-streaming,{dataset},{config}",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
            },
            {
                "id": f"config-parquet-and-info,{dataset},{config}",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
            },
            {
                "id": f"config-parquet,{dataset},{config}",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
            },
            {
                "id": f"config-info,{dataset},{config}",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
            },
            {
                "id": f"/split-names-from-dataset-info,{dataset},{config}",
                "job_state": {"is_in_process": False},
                "cache_state": {
                    "exists": cache_exists,
                    "is_success": cache_exists,
                },  # ^ if this entry is in the cache
            },
            {
                "id": f"config-size,{dataset},{config}",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
            },
        ],
    }


def test_config_state_as_dict() -> None:
    dataset = DATASET_NAME
    config = CONFIG_NAME_1
    upsert_response(
        kind=HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND,
        dataset=DATASET_NAME,
        config=CONFIG_NAME_1,
        split=None,
        content=SPLIT_NAMES_RESPONSE_OK["content"],
        http_status=SPLIT_NAMES_RESPONSE_OK["http_status"],
    )
    processing_graph = PROCESSING_GRAPH
    assert ConfigState(
        dataset=dataset, config=config, processing_graph=processing_graph
    ).as_dict() == get_CONFIG_STATE_DICT(
        dataset=DATASET_NAME,
        config=CONFIG_NAME_1,
        split_states=[
            get_SPLIT_STATE_DICT(dataset=dataset, config=config, split=SPLIT1_NAME),
            get_SPLIT_STATE_DICT(dataset=dataset, config=config, split=SPLIT2_NAME),
        ],
        cache_exists=True,
    )


CONFIG_NAME_2 = "config2"
TWO_CONFIG_NAMES = [CONFIG_NAME_1, CONFIG_NAME_2]
TWO_CONFIG_NAMES_CONTENT_OK = {"config_names": [{"config": config} for config in TWO_CONFIG_NAMES]}
CURRENT_GIT_REVISION = "current_git_revision"


def test_dataset_state_as_dict() -> None:
    dataset = DATASET_NAME
    upsert_response(
        kind=HARD_CODED_CONFIG_NAMES_CACHE_KIND,
        dataset=dataset,
        config=None,
        split=None,
        content=TWO_CONFIG_NAMES_CONTENT_OK,
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind=HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND,
        dataset=dataset,
        config=CONFIG_NAME_1,
        split=None,
        content=SPLIT_NAMES_RESPONSE_OK["content"],
        http_status=SPLIT_NAMES_RESPONSE_OK["http_status"],
    )
    processing_graph = PROCESSING_GRAPH
    assert DatasetState(
        dataset=dataset, processing_graph=processing_graph, revision=CURRENT_GIT_REVISION
    ).as_dict() == {
        "dataset": "dataset",
        "config_states": [
            get_CONFIG_STATE_DICT(
                dataset=dataset,
                config=CONFIG_NAME_1,
                split_states=[
                    get_SPLIT_STATE_DICT(dataset=dataset, config=CONFIG_NAME_1, split=SPLIT1_NAME),
                    get_SPLIT_STATE_DICT(dataset=dataset, config=CONFIG_NAME_1, split=SPLIT2_NAME),
                ],
                cache_exists=True,
            ),
            get_CONFIG_STATE_DICT(
                dataset=dataset,
                config=CONFIG_NAME_2,
                split_states=[],
                cache_exists=False,
            ),
        ],
        "artifact_states": [
            {
                "id": f"/config-names,{DATASET_NAME}",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": True, "is_success": True},  # <- this entry is in the cache
            },
            {
                "id": f"/parquet-and-dataset-info,{DATASET_NAME}",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
            },
            {
                "id": f"dataset-parquet,{DATASET_NAME}",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
            },
            {
                "id": f"dataset-info,{DATASET_NAME}",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
            },
            {
                "id": f"dataset-size,{DATASET_NAME}",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
            },
            {
                "id": f"dataset-split-names-from-streaming,{DATASET_NAME}",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
            },
            {
                "id": f"dataset-split-names-from-dataset-info,{DATASET_NAME}",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
            },
            {
                "id": f"dataset-split-names,{DATASET_NAME}",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
            },
            {
                "id": f"dataset-is-valid,{DATASET_NAME}",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
            },
        ],
    }


CONFIG_PARQUET_AND_INFO_OK = {"config": CONFIG_NAME_1, "content": "not important"}
CONFIG_INFO_OK = {"config": CONFIG_NAME_1, "content": "not important"}


OLD_JOB_RUNNER_VERSION = 1
NEW_JOB_RUNNER_VERSION = 2


def finish_job(job_type: str, content: Any, job_runner_version: Optional[int] = NEW_JOB_RUNNER_VERSION) -> None:
    job_info = Queue().start_job(only_job_types=[job_type])
    upsert_response(
        kind=job_info["type"],
        dataset=job_info["dataset"],
        config=job_info["config"],
        split=job_info["split"],
        content=content,
        http_status=HTTPStatus.OK,
        job_runner_version=job_runner_version,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )
    Queue().finish_job(job_id=job_info["job_id"], finished_status=Status.SUCCESS)


def get_dataset_state(git_revision: Optional[str] = CURRENT_GIT_REVISION) -> DatasetState:
    return DatasetState(dataset=DATASET_NAME, processing_graph=PROCESSING_GRAPH, revision=git_revision)


def assert_dataset_state(
    config_names: List[str],
    split_names_in_first_config: List[str],
    cache_status: Dict[str, List[str]],
    queue_status: Dict[str, List[str]],
    tasks: List[str],
    git_revision: Optional[str] = CURRENT_GIT_REVISION,
) -> DatasetState:
    dataset_state = get_dataset_state(git_revision=git_revision)
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
        # "dataset-is-valid" is also empty, but is marked as blocked because it depends on "dataset-split-names",
        # which is not yet known.
        # No config-level and split-level cache entries is listed, because the config names and splits
        # names are not yet known.
        cache_status={
            "blocked_by_parent": ["dataset-is-valid,dataset"],
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/config-names,dataset",
                "/parquet-and-dataset-info,dataset",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
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
            "CreateJob[/parquet-and-dataset-info,dataset]",
            "CreateJob[dataset-info,dataset]",
            "CreateJob[dataset-parquet,dataset]",
            "CreateJob[dataset-size,dataset]",
            "CreateJob[dataset-split-names,dataset]",
            "CreateJob[dataset-split-names-from-dataset-info,dataset]",
            "CreateJob[dataset-split-names-from-streaming,dataset]",
        ],
    )


def test_plan_job_creation_and_termination() -> None:
    # we launch all the backfill tasks
    dataset_state = get_dataset_state()
    assert dataset_state.plan.as_response() == [
        "CreateJob[/config-names,dataset]",
        "CreateJob[/parquet-and-dataset-info,dataset]",
        "CreateJob[dataset-info,dataset]",
        "CreateJob[dataset-parquet,dataset]",
        "CreateJob[dataset-size,dataset]",
        "CreateJob[dataset-split-names,dataset]",
        "CreateJob[dataset-split-names-from-dataset-info,dataset]",
        "CreateJob[dataset-split-names-from-streaming,dataset]",
    ]
    dataset_state.backfill()
    assert_dataset_state(
        # The config names are not yet known
        config_names=[],
        # The split names are not yet known
        split_names_in_first_config=[],
        # the cache has not changed
        cache_status={
            "blocked_by_parent": ["dataset-is-valid,dataset"],
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/config-names,dataset",
                "/parquet-and-dataset-info,dataset",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [],
        },
        # the jobs have been created and are in process
        queue_status={
            "in_process": [
                "/config-names,dataset",
                "/parquet-and-dataset-info,dataset",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
            ]
        },
        # thus: no new task
        tasks=[],
    )

    # we simulate the job for "/config-names,dataset" has finished
    finish_job("/config-names", TWO_CONFIG_NAMES_CONTENT_OK, job_runner_version=PROCESSING_STEP_CONFIG_NAMES_VERSION)

    assert_dataset_state(
        # The config names are now known
        config_names=TWO_CONFIG_NAMES,
        # The split names are not yet known
        split_names_in_first_config=[],
        # The "/config-names" step is up-to-date
        # Config-level artifacts appear in the "blocked_by_parent" status, because the config names are now known, but
        # the parents are not ready yet.
        # The split-level artifacts are still missing, because the splits names are not yet known, for any config.
        cache_status={
            "blocked_by_parent": [
                "/split-names-from-dataset-info,dataset,config1",
                "/split-names-from-dataset-info,dataset,config2",
                "config-info,dataset,config1",
                "config-info,dataset,config2",
                "config-parquet,dataset,config1",
                "config-parquet,dataset,config2",
                "config-size,dataset,config1",
                "config-size,dataset,config2",
                "dataset-is-valid,dataset",
            ],
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/parquet-and-dataset-info,dataset",
                "/split-names-from-streaming,dataset,config1",
                "/split-names-from-streaming,dataset,config2",
                "config-parquet-and-info,dataset,config1",
                "config-parquet-and-info,dataset,config2",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": ["/config-names,dataset"],
        },
        # the job "/config-names,dataset" is no more in process
        queue_status={
            "in_process": [
                "/parquet-and-dataset-info,dataset",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
            ]
        },
        # The children of "/config-names,dataset" are ready to be backfilled for each config (fan-out)
        tasks=[
            "CreateJob[/split-names-from-streaming,dataset,config1]",
            "CreateJob[/split-names-from-streaming,dataset,config2]",
            "CreateJob[config-parquet-and-info,dataset,config1]",
            "CreateJob[config-parquet-and-info,dataset,config2]",
        ],
    )


def test_plan_only_one_config() -> None:
    # Set the "/config-names,dataset" artifact in cache
    upsert_response(
        kind="/config-names",
        dataset=DATASET_NAME,
        config=None,
        split=None,
        content=TWO_CONFIG_NAMES_CONTENT_OK,
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
        content=TWO_CONFIG_NAMES_CONTENT_OK,
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )

    assert_dataset_state(
        # The config names are known
        config_names=TWO_CONFIG_NAMES,
        # The split names are not yet known
        split_names_in_first_config=[],
        # The children of "config-parquet-and-info" are blocked for config2, but not for config1
        # The two cache entries we created are up-to-date
        cache_status={
            "blocked_by_parent": [
                "/split-names-from-dataset-info,dataset,config1",
                "/split-names-from-dataset-info,dataset,config2",
                "config-info,dataset,config2",
                "config-parquet,dataset,config2",
                "config-size,dataset,config2",
                "dataset-is-valid,dataset",
            ],
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/parquet-and-dataset-info,dataset",
                "/split-names-from-streaming,dataset,config1",
                "/split-names-from-streaming,dataset,config2",
                "config-info,dataset,config1",
                "config-parquet,dataset,config1",
                "config-parquet-and-info,dataset,config2",
                "config-size,dataset,config1",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": ["/config-names,dataset", "config-parquet-and-info,dataset,config1"],
        },
        queue_status={"in_process": []},
        # the children of "config-parquet-and-info" for config1 are ready to be backfilled
        tasks=[
            "CreateJob[/parquet-and-dataset-info,dataset]",
            "CreateJob[/split-names-from-streaming,dataset,config1]",
            "CreateJob[/split-names-from-streaming,dataset,config2]",
            "CreateJob[config-info,dataset,config1]",
            "CreateJob[config-parquet,dataset,config1]",
            "CreateJob[config-parquet-and-info,dataset,config2]",
            "CreateJob[config-size,dataset,config1]",
            "CreateJob[dataset-info,dataset]",
            "CreateJob[dataset-parquet,dataset]",
            "CreateJob[dataset-size,dataset]",
            "CreateJob[dataset-split-names,dataset]",
            "CreateJob[dataset-split-names-from-dataset-info,dataset]",
            "CreateJob[dataset-split-names-from-streaming,dataset]",
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
        content=TWO_CONFIG_NAMES_CONTENT_OK,
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
        job_runner_version=PROCESSING_STEP_CONFIG_NAMES_VERSION,
        error_code=ERROR_CODE_TO_RETRY,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )

    with patch("libcommon.state.ERROR_CODES_TO_RETRY", [ERROR_CODE_TO_RETRY]):
        assert_dataset_state(
            # The config names are known
            config_names=TWO_CONFIG_NAMES,
            # The split names are not yet known
            split_names_in_first_config=[],
            # "/config-names,dataset" is in the cache, but it's not categorized in up to date,
            # but in "cache_is_error_to_retry" due to the error code
            cache_status={
                "blocked_by_parent": [
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
                    "dataset-is-valid,dataset",
                ],
                "cache_has_different_git_revision": [],
                "cache_is_outdated_by_parent": [],
                "cache_is_empty": [
                    "/parquet-and-dataset-info,dataset",
                    "dataset-info,dataset",
                    "dataset-parquet,dataset",
                    "dataset-size,dataset",
                    "dataset-split-names,dataset",
                    "dataset-split-names-from-dataset-info,dataset",
                    "dataset-split-names-from-streaming,dataset",
                ],
                "cache_is_error_to_retry": ["/config-names,dataset"],
                "cache_is_job_runner_obsolete": [],
                "up_to_date": [],
            },
            queue_status={"in_process": []},
            # The "/config-names,dataset" artifact will be retried
            tasks=[
                "CreateJob[/config-names,dataset]",
                "CreateJob[/parquet-and-dataset-info,dataset]",
                "CreateJob[dataset-info,dataset]",
                "CreateJob[dataset-parquet,dataset]",
                "CreateJob[dataset-size,dataset]",
                "CreateJob[dataset-split-names,dataset]",
                "CreateJob[dataset-split-names-from-dataset-info,dataset]",
                "CreateJob[dataset-split-names-from-streaming,dataset]",
            ],
        )


def test_plan_incoherent_state() -> None:
    # Set the "/config-names,dataset" artifact in cache
    upsert_response(
        kind="/config-names",
        dataset=DATASET_NAME,
        config=None,
        split=None,
        content=TWO_CONFIG_NAMES_CONTENT_OK,
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
        content=get_SPLIT_NAMES_CONTENT_OK(dataset=DATASET_NAME, config=CONFIG_NAME_1, splits=SPLIT_NAMES_OK),
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_SPLIT_NAMES_FROM_DATASET_INFO_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )

    assert_dataset_state(
        # The config names are known
        config_names=TWO_CONFIG_NAMES,
        # The split names are known (even if it's from an outdated artifact)
        split_names_in_first_config=SPLIT_NAMES_OK,
        # "/split-names-from-dataset-info,dataset,config1" is blocked by its parents, even if it's in cache, because
        # two ancestors are missing. It's not listed in up-to-date, even if it has been cached after /config-names.
        cache_status={
            "blocked_by_parent": [
                "/split-names-from-dataset-info,dataset,config1",
                "/split-names-from-dataset-info,dataset,config2",
                "config-info,dataset,config1",
                "config-info,dataset,config2",
                "config-parquet,dataset,config1",
                "config-parquet,dataset,config2",
                "config-size,dataset,config1",
                "config-size,dataset,config2",
                "dataset-is-valid,dataset",
                "split-first-rows-from-parquet,dataset,config1,split1",
                "split-first-rows-from-parquet,dataset,config1,split2",
                "split-first-rows-from-streaming,dataset,config1,split1",
                "split-first-rows-from-streaming,dataset,config1,split2",
            ],
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/parquet-and-dataset-info,dataset",
                "/split-names-from-streaming,dataset,config1",
                "/split-names-from-streaming,dataset,config2",
                "config-parquet-and-info,dataset,config1",
                "config-parquet-and-info,dataset,config2",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": ["/config-names,dataset"],
        },
        queue_status={"in_process": []},
        # The children of "/split-names-from-dataset-info,dataset,config1" are not ready to be backfilled
        # because it's blocked by its parents.
        tasks=[
            "CreateJob[/parquet-and-dataset-info,dataset]",
            "CreateJob[/split-names-from-streaming,dataset,config1]",
            "CreateJob[/split-names-from-streaming,dataset,config2]",
            "CreateJob[config-parquet-and-info,dataset,config1]",
            "CreateJob[config-parquet-and-info,dataset,config2]",
            "CreateJob[dataset-info,dataset]",
            "CreateJob[dataset-parquet,dataset]",
            "CreateJob[dataset-size,dataset]",
            "CreateJob[dataset-split-names,dataset]",
            "CreateJob[dataset-split-names-from-dataset-info,dataset]",
            "CreateJob[dataset-split-names-from-streaming,dataset]",
        ],
    )


def test_plan_get_splits() -> None:
    # Set the "/config-names,dataset" artifact in cache
    upsert_response(
        kind="/config-names",
        dataset=DATASET_NAME,
        config=None,
        split=None,
        content=TWO_CONFIG_NAMES_CONTENT_OK,
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
        content=TWO_CONFIG_NAMES_CONTENT_OK,  # <- not important
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )
    # Set the "config-info,dataset,config1" artifact in cache
    upsert_response(
        kind="config-info",
        dataset=DATASET_NAME,
        config=CONFIG_NAME_1,
        split=None,
        content=TWO_CONFIG_NAMES_CONTENT_OK,  # <- not important
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_INFO_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )
    # Set the "/split-names-from-dataset-info,dataset,config1" artifact in cache
    upsert_response(
        kind="/split-names-from-dataset-info",
        dataset=DATASET_NAME,
        config=CONFIG_NAME_1,
        split=None,
        content=get_SPLIT_NAMES_CONTENT_OK(dataset=DATASET_NAME, config=CONFIG_NAME_1, splits=SPLIT_NAMES_OK),
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_SPLIT_NAMES_FROM_DATASET_INFO_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )
    # Set the "/split-names-from-streaming,dataset,config1" artifact in cache
    upsert_response(
        kind="/split-names-from-streaming",
        dataset=DATASET_NAME,
        config=CONFIG_NAME_1,
        split=None,
        content=get_SPLIT_NAMES_CONTENT_OK(dataset=DATASET_NAME, config=CONFIG_NAME_1, splits=SPLIT_NAMES_OK),
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_SPLIT_NAMES_FROM_STREAMING_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )

    assert_dataset_state(
        # The config names are known
        config_names=TWO_CONFIG_NAMES,
        # The split names are known (even if it's from an outdated artifact)
        split_names_in_first_config=SPLIT_NAMES_OK,
        # The chains /config-names,dataset -> config-parquet-and-info,dataset,config1 -> config-info,dataset,config1
        # -> /split-names-from-dataset-info,dataset,config1 and
        # /config-names,dataset -> /split-names-from-streaming,dataset,config1 are up to date
        cache_status={
            "blocked_by_parent": [
                "/split-names-from-dataset-info,dataset,config2",
                "config-info,dataset,config2",
                "config-parquet,dataset,config2",
                "config-size,dataset,config2",
                "dataset-is-valid,dataset",
                "split-first-rows-from-parquet,dataset,config1,split1",
                "split-first-rows-from-parquet,dataset,config1,split2",
            ],
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/parquet-and-dataset-info,dataset",
                "/split-names-from-streaming,dataset,config2",
                "config-parquet,dataset,config1",
                "config-parquet-and-info,dataset,config2",
                "config-size,dataset,config1",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
                "split-first-rows-from-streaming,dataset,config1,split1",
                "split-first-rows-from-streaming,dataset,config1,split2",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [
                "/config-names,dataset",
                "/split-names-from-dataset-info,dataset,config1",
                "/split-names-from-streaming,dataset,config1",
                "config-info,dataset,config1",
                "config-parquet-and-info,dataset,config1",
            ],
        },
        queue_status={"in_process": []},
        # split-first-rows-from-streaming is now ready to be backfilled for split1 and split2 of dataset,config1
        tasks=[
            "CreateJob[/parquet-and-dataset-info,dataset]",
            "CreateJob[/split-names-from-streaming,dataset,config2]",
            "CreateJob[config-parquet,dataset,config1]",
            "CreateJob[config-parquet-and-info,dataset,config2]",
            "CreateJob[config-size,dataset,config1]",
            "CreateJob[dataset-info,dataset]",
            "CreateJob[dataset-parquet,dataset]",
            "CreateJob[dataset-size,dataset]",
            "CreateJob[dataset-split-names,dataset]",
            "CreateJob[dataset-split-names-from-dataset-info,dataset]",
            "CreateJob[dataset-split-names-from-streaming,dataset]",
            "CreateJob[split-first-rows-from-streaming,dataset,config1,split1]",
            "CreateJob[split-first-rows-from-streaming,dataset,config1,split2]",
        ],
    )


def test_plan_updated_at() -> None:
    # Set the "/config-names,dataset" artifact in cache
    upsert_response(
        kind="/config-names",
        dataset=DATASET_NAME,
        config=None,
        split=None,
        content=TWO_CONFIG_NAMES_CONTENT_OK,
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
        content=TWO_CONFIG_NAMES_CONTENT_OK,  # <- not important
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )
    # Set the "config-info,dataset,config1" artifact in cache
    upsert_response(
        kind="config-info",
        dataset=DATASET_NAME,
        config=CONFIG_NAME_1,
        split=None,
        content=TWO_CONFIG_NAMES_CONTENT_OK,  # <- not important
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_INFO_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )
    # Set the "/split-names-from-dataset-info,dataset,config1" artifact in cache
    upsert_response(
        kind="/split-names-from-dataset-info",
        dataset=DATASET_NAME,
        config=CONFIG_NAME_1,
        split=None,
        content=get_SPLIT_NAMES_CONTENT_OK(dataset=DATASET_NAME, config=CONFIG_NAME_1, splits=SPLIT_NAMES_OK),
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_SPLIT_NAMES_FROM_DATASET_INFO_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )
    # Set the "/split-names-from-streaming,dataset,config1" artifact in cache
    upsert_response(
        kind="/split-names-from-streaming",
        dataset=DATASET_NAME,
        config=CONFIG_NAME_1,
        split=None,
        content=get_SPLIT_NAMES_CONTENT_OK(dataset=DATASET_NAME, config=CONFIG_NAME_1, splits=SPLIT_NAMES_OK),
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_SPLIT_NAMES_FROM_STREAMING_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )

    # Now: refresh again the "/config-names,dataset" artifact in cache
    upsert_response(
        kind="/config-names",
        dataset=DATASET_NAME,
        config=None,
        split=None,
        content=TWO_CONFIG_NAMES_CONTENT_OK,
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_NAMES_VERSION,
        dataset_git_revision=CURRENT_GIT_REVISION,
    )

    assert_dataset_state(
        # The config names are known
        config_names=TWO_CONFIG_NAMES,
        # The split names are known (even if it's from an outdated artifact)
        split_names_in_first_config=SPLIT_NAMES_OK,
        # config-parquet-and-info,dataset,config1 is marked as outdate by parent,
        # as well as /split-names-from-streaming,dataset,config1
        # This means that all their children are marked as blocked by parent
        # Only "/config-names,dataset" is marked as up to date
        cache_status={
            "blocked_by_parent": [
                "/split-names-from-dataset-info,dataset,config1",
                "/split-names-from-dataset-info,dataset,config2",
                "config-info,dataset,config1",
                "config-info,dataset,config2",
                "config-parquet,dataset,config1",
                "config-parquet,dataset,config2",
                "config-size,dataset,config1",
                "config-size,dataset,config2",
                "dataset-is-valid,dataset",
                "split-first-rows-from-parquet,dataset,config1,split1",
                "split-first-rows-from-parquet,dataset,config1,split2",
                "split-first-rows-from-streaming,dataset,config1,split1",
                "split-first-rows-from-streaming,dataset,config1,split2",
            ],
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [
                "/split-names-from-streaming,dataset,config1",
                "config-parquet-and-info,dataset,config1",
            ],
            "cache_is_empty": [
                "/parquet-and-dataset-info,dataset",
                "/split-names-from-streaming,dataset,config2",
                "config-parquet-and-info,dataset,config2",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": ["/config-names,dataset"],
        },
        queue_status={"in_process": []},
        # config-parquet-and-info and /split-names-from-streaming will be refreshed for both configs
        # (even if config1 is already in the cache -> but it's outdated)
        tasks=[
            "CreateJob[/parquet-and-dataset-info,dataset]",
            "CreateJob[/split-names-from-streaming,dataset,config1]",
            "CreateJob[/split-names-from-streaming,dataset,config2]",
            "CreateJob[config-parquet-and-info,dataset,config1]",
            "CreateJob[config-parquet-and-info,dataset,config2]",
            "CreateJob[dataset-info,dataset]",
            "CreateJob[dataset-parquet,dataset]",
            "CreateJob[dataset-size,dataset]",
            "CreateJob[dataset-split-names,dataset]",
            "CreateJob[dataset-split-names-from-dataset-info,dataset]",
            "CreateJob[dataset-split-names-from-streaming,dataset]",
        ],
    )


def test_plan_job_runner_version() -> None:
    # Set the "/config-names,dataset" artifact in cache
    upsert_response(
        kind="/config-names",
        dataset=DATASET_NAME,
        config=None,
        split=None,
        content=TWO_CONFIG_NAMES_CONTENT_OK,
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
        content=TWO_CONFIG_NAMES_CONTENT_OK,  # <- not important
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
        content=TWO_CONFIG_NAMES_CONTENT_OK,  # <- not important
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_PARQUET_VERSION - 1,  # <- current version is 3, this one is 1
        dataset_git_revision=CURRENT_GIT_REVISION,
    )

    assert_dataset_state(
        # The config names are known
        config_names=TWO_CONFIG_NAMES,
        # The split names are not known
        split_names_in_first_config=[],
        # config-parquet,dataset,config1 is in the category: "is_job_runner_obsolete"
        cache_status={
            "blocked_by_parent": [
                "/split-names-from-dataset-info,dataset,config1",
                "/split-names-from-dataset-info,dataset,config2",
                "config-info,dataset,config2",
                "config-parquet,dataset,config2",
                "config-size,dataset,config2",
                "dataset-is-valid,dataset",
            ],
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/parquet-and-dataset-info,dataset",
                "/split-names-from-streaming,dataset,config1",
                "/split-names-from-streaming,dataset,config2",
                "config-info,dataset,config1",
                "config-parquet-and-info,dataset,config2",
                "config-size,dataset,config1",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": ["config-parquet,dataset,config1"],
            "up_to_date": ["/config-names,dataset", "config-parquet-and-info,dataset,config1"],
        },
        queue_status={"in_process": []},
        # config-parquet,dataset,config1 will be refreshed because its job runner has been upgraded
        tasks=[
            "CreateJob[/parquet-and-dataset-info,dataset]",
            "CreateJob[/split-names-from-streaming,dataset,config1]",
            "CreateJob[/split-names-from-streaming,dataset,config2]",
            "CreateJob[config-info,dataset,config1]",
            "CreateJob[config-parquet,dataset,config1]",
            "CreateJob[config-parquet-and-info,dataset,config2]",
            "CreateJob[config-size,dataset,config1]",
            "CreateJob[dataset-info,dataset]",
            "CreateJob[dataset-parquet,dataset]",
            "CreateJob[dataset-size,dataset]",
            "CreateJob[dataset-split-names,dataset]",
            "CreateJob[dataset-split-names-from-dataset-info,dataset]",
            "CreateJob[dataset-split-names-from-streaming,dataset]",
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
        content=TWO_CONFIG_NAMES_CONTENT_OK,
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_NAMES_VERSION,
        dataset_git_revision=cached_dataset_get_revision,
    )

    if expect_refresh:
        # if the git revision is different from the current dataset git revision, the artifact will be refreshed
        # and its children will be blocked
        assert_dataset_state(
            git_revision=dataset_git_revision,
            # The config names are known
            config_names=TWO_CONFIG_NAMES,
            # The split names are not known
            split_names_in_first_config=[],
            cache_status={
                "blocked_by_parent": [
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
                    "dataset-is-valid,dataset",
                ],
                "cache_has_different_git_revision": ["/config-names,dataset"],
                "cache_is_outdated_by_parent": [],
                "cache_is_empty": [
                    "/parquet-and-dataset-info,dataset",
                    "dataset-info,dataset",
                    "dataset-parquet,dataset",
                    "dataset-size,dataset",
                    "dataset-split-names,dataset",
                    "dataset-split-names-from-dataset-info,dataset",
                    "dataset-split-names-from-streaming,dataset",
                ],
                "cache_is_error_to_retry": [],
                "cache_is_job_runner_obsolete": [],
                "up_to_date": [],
            },
            queue_status={"in_process": []},
            tasks=[
                "CreateJob[/config-names,dataset]",
                "CreateJob[/parquet-and-dataset-info,dataset]",
                "CreateJob[dataset-info,dataset]",
                "CreateJob[dataset-parquet,dataset]",
                "CreateJob[dataset-size,dataset]",
                "CreateJob[dataset-split-names,dataset]",
                "CreateJob[dataset-split-names-from-dataset-info,dataset]",
                "CreateJob[dataset-split-names-from-streaming,dataset]",
            ],
        )
    else:
        assert_dataset_state(
            git_revision=dataset_git_revision,
            # The config names are known
            config_names=TWO_CONFIG_NAMES,
            # The split names are not known
            split_names_in_first_config=[],
            cache_status={
                "blocked_by_parent": [
                    "/split-names-from-dataset-info,dataset,config1",
                    "/split-names-from-dataset-info,dataset,config2",
                    "config-info,dataset,config1",
                    "config-info,dataset,config2",
                    "config-parquet,dataset,config1",
                    "config-parquet,dataset,config2",
                    "config-size,dataset,config1",
                    "config-size,dataset,config2",
                    "dataset-is-valid,dataset",
                ],
                "cache_has_different_git_revision": [],
                "cache_is_outdated_by_parent": [],
                "cache_is_empty": [
                    "/parquet-and-dataset-info,dataset",
                    "/split-names-from-streaming,dataset,config1",
                    "/split-names-from-streaming,dataset,config2",
                    "config-parquet-and-info,dataset,config1",
                    "config-parquet-and-info,dataset,config2",
                    "dataset-info,dataset",
                    "dataset-parquet,dataset",
                    "dataset-size,dataset",
                    "dataset-split-names,dataset",
                    "dataset-split-names-from-dataset-info,dataset",
                    "dataset-split-names-from-streaming,dataset",
                ],
                "cache_is_error_to_retry": [],
                "cache_is_job_runner_obsolete": [],
                "up_to_date": ["/config-names,dataset"],
            },
            queue_status={"in_process": []},
            tasks=[
                "CreateJob[/parquet-and-dataset-info,dataset]",
                "CreateJob[/split-names-from-streaming,dataset,config1]",
                "CreateJob[/split-names-from-streaming,dataset,config2]",
                "CreateJob[config-parquet-and-info,dataset,config1]",
                "CreateJob[config-parquet-and-info,dataset,config2]",
                "CreateJob[dataset-info,dataset]",
                "CreateJob[dataset-parquet,dataset]",
                "CreateJob[dataset-size,dataset]",
                "CreateJob[dataset-split-names,dataset]",
                "CreateJob[dataset-split-names-from-dataset-info,dataset]",
                "CreateJob[dataset-split-names-from-streaming,dataset]",
            ],
        )
