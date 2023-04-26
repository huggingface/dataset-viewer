# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, List, Mapping, Optional, TypedDict

import pytest

from libcommon.config import ProcessingGraphConfig
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
    return {"splits": [{"dataset": dataset, "config": config, "split": split_name} for split_name in splits]}


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
            {
                "id": f"split-opt-in-out-urls-scan,{dataset},{config},{split}",
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
