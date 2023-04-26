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
    ArtifactState,
    CacheState,
    ConfigState,
    DatasetState,
    JobState,
    SplitState,
    fetch_names,
)

from .utils import (
    CONFIG_NAME_1,
    CONFIG_NAME_2,
    CONFIG_NAMES,
    CONFIG_NAMES_CONTENT,
    CURRENT_GIT_REVISION,
    DATASET_NAME,
    SPLIT_NAME_1,
    SPLIT_NAME_2,
    SPLIT_NAMES,
    SPLIT_NAMES_CONTENT,
)


class ResponseSpec(TypedDict):
    content: Mapping[str, Any]
    http_status: HTTPStatus


CACHE_KIND = "cache_kind"
JOB_TYPE = "job_type"
SPLIT_NAMES_RESPONSE_OK = ResponseSpec(content=SPLIT_NAMES_CONTENT, http_status=HTTPStatus.OK)
CONTENT_ERROR = {"error": "error"}
RESPONSE_ERROR = ResponseSpec(content=CONTENT_ERROR, http_status=HTTPStatus.INTERNAL_SERVER_ERROR)


PROCESSING_GRAPH = ProcessingGraph(processing_graph_specification=ProcessingGraphConfig().specification)


@pytest.fixture(autouse=True)
def queue_mongo_resource_autouse(queue_mongo_resource: QueueMongoResource) -> QueueMongoResource:
    return queue_mongo_resource


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(cache_mongo_resource: CacheMongoResource) -> CacheMongoResource:
    return cache_mongo_resource


CACHE_KIND_A = "cache_kind_a"
CACHE_KIND_B = "cache_kind_b"
NAMES_FIELD = "names"
NAME_FIELD = "name"
NAMES = ["name_1", "name_2", "name_3"]
NAMES_RESPONSE_OK = ResponseSpec(
    content={NAMES_FIELD: [{NAME_FIELD: name} for name in NAMES]}, http_status=HTTPStatus.OK
)


@pytest.mark.parametrize(
    "cache_kinds,response_spec_by_kind,expected_names",
    [
        ([], {}, None),
        ([CACHE_KIND_A], {}, None),
        ([CACHE_KIND_A], {CACHE_KIND_A: RESPONSE_ERROR}, None),
        ([CACHE_KIND_A], {CACHE_KIND_A: NAMES_RESPONSE_OK}, NAMES),
        ([CACHE_KIND_A, CACHE_KIND_B], {CACHE_KIND_A: NAMES_RESPONSE_OK}, NAMES),
        ([CACHE_KIND_A, CACHE_KIND_B], {CACHE_KIND_A: NAMES_RESPONSE_OK, CACHE_KIND_B: RESPONSE_ERROR}, NAMES),
        ([CACHE_KIND_A, CACHE_KIND_B], {CACHE_KIND_A: NAMES_RESPONSE_OK, CACHE_KIND_B: NAMES_RESPONSE_OK}, NAMES),
        ([CACHE_KIND_A, CACHE_KIND_B], {CACHE_KIND_A: RESPONSE_ERROR, CACHE_KIND_B: RESPONSE_ERROR}, None),
    ],
)
def test_fetch_names(
    cache_kinds: List[str],
    response_spec_by_kind: Mapping[str, Mapping[str, Any]],
    expected_names: Optional[List[str]],
) -> None:
    raises = expected_names is None
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
            fetch_names(
                dataset=DATASET_NAME,
                config=CONFIG_NAME_1,
                cache_kinds=cache_kinds,
                names_field=NAMES_FIELD,
                name_field=NAME_FIELD,
            )
    else:
        names = fetch_names(
            dataset=DATASET_NAME,
            config=CONFIG_NAME_1,
            cache_kinds=cache_kinds,
            names_field=NAMES_FIELD,
            name_field=NAME_FIELD,
        )
        assert names == expected_names


@pytest.mark.parametrize(
    "dataset,config,split,job_type",
    [
        (DATASET_NAME, None, None, JOB_TYPE),
        (DATASET_NAME, CONFIG_NAME_1, None, JOB_TYPE),
        (DATASET_NAME, CONFIG_NAME_1, SPLIT_NAME_1, JOB_TYPE),
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
        (DATASET_NAME, CONFIG_NAME_1, SPLIT_NAME_1, JOB_TYPE),
    ],
)
def test_job_state_as_dict(dataset: str, config: Optional[str], split: Optional[str], job_type: str) -> None:
    queue = Queue()
    queue.upsert_job(job_type=job_type, dataset=dataset, config=config, split=split)
    assert JobState(dataset=dataset, config=config, split=split, job_type=job_type).as_dict() == {
        "is_in_process": True,
    }


@pytest.mark.parametrize(
    "dataset,config,split,cache_kind",
    [
        (DATASET_NAME, None, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME_1, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME_1, SPLIT_NAME_1, CACHE_KIND),
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
        (DATASET_NAME, CONFIG_NAME_1, SPLIT_NAME_1, CACHE_KIND),
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
        (DATASET_NAME, CONFIG_NAME_1, SPLIT_NAME_1, CACHE_KIND),
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


def test_split_state_as_dict() -> None:
    dataset = DATASET_NAME
    config = CONFIG_NAME_1
    split = SPLIT_NAME_1
    processing_graph = PROCESSING_GRAPH
    assert SplitState(
        dataset=dataset, config=config, split=split, processing_graph=processing_graph
    ).as_dict() == get_SPLIT_STATE_DICT(dataset=dataset, config=config, split=split)


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
            get_SPLIT_STATE_DICT(dataset=dataset, config=config, split=SPLIT_NAME_1),
            get_SPLIT_STATE_DICT(dataset=dataset, config=config, split=SPLIT_NAME_2),
        ],
        cache_exists=True,
    )


def test_dataset_state_as_dict() -> None:
    dataset = DATASET_NAME
    upsert_response(
        kind=HARD_CODED_CONFIG_NAMES_CACHE_KIND,
        dataset=dataset,
        config=None,
        split=None,
        content=CONFIG_NAMES_CONTENT,
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
                    get_SPLIT_STATE_DICT(dataset=dataset, config=CONFIG_NAME_1, split=SPLIT_NAME_1),
                    get_SPLIT_STATE_DICT(dataset=dataset, config=CONFIG_NAME_1, split=SPLIT_NAME_2),
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
