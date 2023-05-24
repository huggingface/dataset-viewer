# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, List, Mapping, Optional, TypedDict

import pytest

from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import (
    delete_response,
    get_cache_entries_df,
    upsert_response,
)
from libcommon.state import (
    ArtifactState,
    CacheState,
    ConfigState,
    DatasetState,
    SplitState,
    fetch_names,
)

from .utils import (
    CONFIG_NAME_1,
    CONFIG_NAMES,
    CONFIG_NAMES_CONTENT,
    DATASET_NAME,
    REVISION_NAME,
    SPLIT_NAME_1,
    SPLIT_NAMES,
    SPLIT_NAMES_CONTENT,
)


class ResponseSpec(TypedDict):
    content: Mapping[str, Any]
    http_status: HTTPStatus


CACHE_KIND = "cache_kind"
CACHE_KIND_A = "cache_kind_a"
CACHE_KIND_B = "cache_kind_b"
CONTENT_ERROR = {"error": "error"}
JOB_TYPE = "job_type"
NAME_FIELD = "name"
NAMES = ["name_1", "name_2", "name_3"]
NAMES_FIELD = "names"
NAMES_RESPONSE_OK = ResponseSpec(
    content={NAMES_FIELD: [{NAME_FIELD: name} for name in NAMES]}, http_status=HTTPStatus.OK
)
STEP_DATASET_A = "dataset-a"
STEP_CONFIG_B = "config-b"
STEP_SPLIT_C = "split-c"
PROCESSING_GRAPH = ProcessingGraph(
    processing_graph_specification={
        STEP_DATASET_A: {"input_type": "dataset", "provides_dataset_config_names": True},
        STEP_CONFIG_B: {"input_type": "config", "provides_config_split_names": True, "triggered_by": STEP_DATASET_A},
        STEP_SPLIT_C: {"input_type": "split", "triggered_by": STEP_CONFIG_B},
    }
)
RESPONSE_ERROR = ResponseSpec(content=CONTENT_ERROR, http_status=HTTPStatus.INTERNAL_SERVER_ERROR)


@pytest.fixture(autouse=True)
def queue_mongo_resource_autouse(queue_mongo_resource: QueueMongoResource) -> QueueMongoResource:
    return queue_mongo_resource


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(cache_mongo_resource: CacheMongoResource) -> CacheMongoResource:
    return cache_mongo_resource


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
    "dataset,config,split,cache_kind",
    [
        (DATASET_NAME, None, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME_1, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME_1, SPLIT_NAME_1, CACHE_KIND),
    ],
)
def test_cache_state_exists(dataset: str, config: Optional[str], split: Optional[str], cache_kind: str) -> None:
    assert not CacheState(
        dataset=dataset,
        config=config,
        split=split,
        cache_kind=cache_kind,
        cache_entries_df=get_cache_entries_df(dataset=dataset),
    ).exists
    upsert_response(
        kind=cache_kind, dataset=dataset, config=config, split=split, content={}, http_status=HTTPStatus.OK
    )
    assert CacheState(
        dataset=dataset,
        config=config,
        split=split,
        cache_kind=cache_kind,
        cache_entries_df=get_cache_entries_df(dataset=dataset),
    ).exists
    delete_response(kind=cache_kind, dataset=dataset, config=config, split=split)
    assert not CacheState(
        dataset=dataset,
        config=config,
        split=split,
        cache_kind=cache_kind,
        cache_entries_df=get_cache_entries_df(dataset=dataset),
    ).exists


@pytest.mark.parametrize(
    "dataset,config,split,cache_kind",
    [
        (DATASET_NAME, None, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME_1, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME_1, SPLIT_NAME_1, CACHE_KIND),
    ],
)
def test_cache_state_is_success(dataset: str, config: Optional[str], split: Optional[str], cache_kind: str) -> None:
    assert not CacheState(
        dataset=dataset,
        config=config,
        split=split,
        cache_kind=cache_kind,
        cache_entries_df=get_cache_entries_df(dataset=dataset),
    ).is_success
    upsert_response(
        kind=cache_kind, dataset=dataset, config=config, split=split, content={}, http_status=HTTPStatus.OK
    )
    assert CacheState(
        dataset=dataset,
        config=config,
        split=split,
        cache_kind=cache_kind,
        cache_entries_df=get_cache_entries_df(dataset=dataset),
    ).is_success
    upsert_response(
        kind=cache_kind,
        dataset=dataset,
        config=config,
        split=split,
        content={},
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    )
    assert not CacheState(
        dataset=dataset,
        config=config,
        split=split,
        cache_kind=cache_kind,
        cache_entries_df=get_cache_entries_df(dataset=dataset),
    ).is_success
    delete_response(kind=cache_kind, dataset=dataset, config=config, split=split)
    assert not CacheState(
        dataset=dataset,
        config=config,
        split=split,
        cache_kind=cache_kind,
        cache_entries_df=get_cache_entries_df(dataset=dataset),
    ).is_success


def test_artifact_state() -> None:
    dataset = DATASET_NAME
    revision = REVISION_NAME
    config = None
    split = None
    processing_step_name = "dataset-a"
    processing_step = PROCESSING_GRAPH.get_processing_step(processing_step_name)
    artifact_state = ArtifactState(
        dataset=dataset,
        revision=revision,
        config=config,
        split=split,
        processing_step=processing_step,
        pending_jobs_df=Queue().get_pending_jobs_df(dataset=dataset, revision=revision),
        cache_entries_df=get_cache_entries_df(dataset=dataset),
    )
    assert artifact_state.id == f"{processing_step_name},{dataset},{revision}"
    assert not artifact_state.cache_state.exists
    assert not artifact_state.cache_state.is_success
    assert not artifact_state.job_state.is_in_process


def test_split_state() -> None:
    dataset = DATASET_NAME
    revision = REVISION_NAME
    config = CONFIG_NAME_1
    split = SPLIT_NAME_1
    expected_split_processing_step_name = "split-c"
    split_state = SplitState(
        dataset=dataset,
        revision=revision,
        config=config,
        split=split,
        processing_graph=PROCESSING_GRAPH,
        pending_jobs_df=Queue()._get_df(jobs=[]),
        cache_entries_df=get_cache_entries_df(dataset=dataset),
    )

    assert split_state.dataset == dataset
    assert split_state.revision == revision
    assert split_state.config == config
    assert split_state.split == split

    assert len(split_state.artifact_state_by_step) == 1
    assert expected_split_processing_step_name in split_state.artifact_state_by_step
    artifact_state = split_state.artifact_state_by_step[expected_split_processing_step_name]
    assert artifact_state.id == f"{expected_split_processing_step_name},{dataset},{revision},{config},{split}"
    assert not artifact_state.cache_state.exists
    assert not artifact_state.cache_state.is_success
    assert not artifact_state.job_state.is_in_process


def test_config_state_as_dict() -> None:
    dataset = DATASET_NAME
    revision = REVISION_NAME
    config = CONFIG_NAME_1
    expected_config_processing_step_name = "config-b"
    processing_step = PROCESSING_GRAPH.get_processing_step(expected_config_processing_step_name)

    upsert_response(
        kind=processing_step.cache_kind,
        dataset=dataset,
        config=config,
        split=None,
        content=SPLIT_NAMES_CONTENT,
        http_status=HTTPStatus.OK,
    )
    config_state = ConfigState(
        dataset=dataset,
        revision=revision,
        config=config,
        processing_graph=PROCESSING_GRAPH,
        pending_jobs_df=Queue()._get_df(jobs=[]),
        cache_entries_df=get_cache_entries_df(dataset=dataset),
    )

    assert config_state.dataset == dataset
    assert config_state.revision == revision
    assert config_state.config == config

    assert len(config_state.artifact_state_by_step) == 1
    assert expected_config_processing_step_name in config_state.artifact_state_by_step
    artifact_state = config_state.artifact_state_by_step[expected_config_processing_step_name]
    assert artifact_state.id == f"{expected_config_processing_step_name},{dataset},{revision},{config}"
    assert artifact_state.cache_state.exists  # <- in the cache
    assert artifact_state.cache_state.is_success  # <- is a success
    assert not artifact_state.job_state.is_in_process

    assert config_state.split_names == SPLIT_NAMES
    assert len(config_state.split_states) == len(SPLIT_NAMES)
    assert config_state.split_states[0].split == SPLIT_NAMES[0]
    assert config_state.split_states[1].split == SPLIT_NAMES[1]


def test_dataset_state_as_dict() -> None:
    dataset = DATASET_NAME
    revision = REVISION_NAME
    expected_dataset_processing_step_name = "dataset-a"
    dataset_step = PROCESSING_GRAPH.get_processing_step(expected_dataset_processing_step_name)
    expected_config_processing_step_name = "config-b"
    config_step = PROCESSING_GRAPH.get_processing_step(expected_config_processing_step_name)
    upsert_response(
        kind=dataset_step.cache_kind,
        dataset=dataset,
        config=None,
        split=None,
        content=CONFIG_NAMES_CONTENT,
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind=config_step.cache_kind,
        dataset=dataset,
        config=CONFIG_NAME_1,
        split=None,
        content=SPLIT_NAMES_CONTENT,
        http_status=HTTPStatus.OK,
    )
    dataset_state = DatasetState(dataset=dataset, revision=revision, processing_graph=PROCESSING_GRAPH)

    assert dataset_state.dataset == dataset
    assert dataset_state.revision == revision

    assert len(dataset_state.artifact_state_by_step) == 1
    assert expected_dataset_processing_step_name in dataset_state.artifact_state_by_step
    artifact_state = dataset_state.artifact_state_by_step[expected_dataset_processing_step_name]
    assert artifact_state.id == f"{expected_dataset_processing_step_name},{dataset},{revision}"
    assert artifact_state.cache_state.exists  # <- in the cache
    assert artifact_state.cache_state.is_success  # <- is a success
    assert not artifact_state.job_state.is_in_process

    assert dataset_state.config_names == CONFIG_NAMES
    assert len(dataset_state.config_states) == len(CONFIG_NAMES)
    assert dataset_state.config_states[0].config == CONFIG_NAMES[0]
    assert dataset_state.config_states[1].config == CONFIG_NAMES[1]
