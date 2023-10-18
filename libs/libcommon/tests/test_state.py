# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus
from typing import Optional

import pytest

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
)

from .utils import (
    CACHE_KIND,
    CONFIG_NAME_1,
    CONFIG_NAMES,
    CONFIG_NAMES_CONTENT,
    DATASET_NAME,
    JOB_RUNNER_VERSION,
    PROCESSING_GRAPH,
    REVISION_NAME,
    SPLIT_NAME_1,
    SPLIT_NAMES,
    SPLIT_NAMES_CONTENT,
)


@pytest.fixture(autouse=True)
def queue_mongo_resource_autouse(queue_mongo_resource: QueueMongoResource) -> QueueMongoResource:
    return queue_mongo_resource


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(cache_mongo_resource: CacheMongoResource) -> CacheMongoResource:
    return cache_mongo_resource


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
        job_runner_version=JOB_RUNNER_VERSION,
    ).exists
    upsert_response(
        kind=cache_kind,
        dataset=dataset,
        config=config,
        split=split,
        content={},
        http_status=HTTPStatus.OK,
        dataset_git_revision=REVISION_NAME,
    )
    assert CacheState(
        dataset=dataset,
        config=config,
        split=split,
        cache_kind=cache_kind,
        cache_entries_df=get_cache_entries_df(dataset=dataset),
        job_runner_version=JOB_RUNNER_VERSION,
    ).exists
    delete_response(kind=cache_kind, dataset=dataset, config=config, split=split)
    assert not CacheState(
        dataset=dataset,
        config=config,
        split=split,
        cache_kind=cache_kind,
        cache_entries_df=get_cache_entries_df(dataset=dataset),
        job_runner_version=JOB_RUNNER_VERSION,
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
        job_runner_version=JOB_RUNNER_VERSION,
    ).is_success
    upsert_response(
        kind=cache_kind,
        dataset=dataset,
        config=config,
        split=split,
        content={},
        http_status=HTTPStatus.OK,
        dataset_git_revision=REVISION_NAME,
    )
    assert CacheState(
        dataset=dataset,
        config=config,
        split=split,
        cache_kind=cache_kind,
        cache_entries_df=get_cache_entries_df(dataset=dataset),
        job_runner_version=JOB_RUNNER_VERSION,
    ).is_success
    upsert_response(
        kind=cache_kind,
        dataset=dataset,
        config=config,
        split=split,
        content={},
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
        dataset_git_revision=REVISION_NAME,
    )
    assert not CacheState(
        dataset=dataset,
        config=config,
        split=split,
        cache_kind=cache_kind,
        cache_entries_df=get_cache_entries_df(dataset=dataset),
        job_runner_version=JOB_RUNNER_VERSION,
    ).is_success
    delete_response(kind=cache_kind, dataset=dataset, config=config, split=split)
    assert not CacheState(
        dataset=dataset,
        config=config,
        split=split,
        cache_kind=cache_kind,
        cache_entries_df=get_cache_entries_df(dataset=dataset),
        job_runner_version=JOB_RUNNER_VERSION,
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
        pending_jobs_df=Queue().get_pending_jobs_df(dataset=dataset),
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
        dataset_git_revision=REVISION_NAME,
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
        dataset_git_revision=REVISION_NAME,
    )
    upsert_response(
        kind=config_step.cache_kind,
        dataset=dataset,
        config=CONFIG_NAME_1,
        split=None,
        content=SPLIT_NAMES_CONTENT,
        http_status=HTTPStatus.OK,
        dataset_git_revision=REVISION_NAME,
    )
    dataset_state = DatasetState(
        dataset=dataset,
        revision=revision,
        processing_graph=PROCESSING_GRAPH,
        pending_jobs_df=Queue()._get_df(jobs=[]),
        cache_entries_df=get_cache_entries_df(dataset=dataset),
    )

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
