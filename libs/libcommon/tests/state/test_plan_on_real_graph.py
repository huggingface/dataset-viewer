# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus

import pytest

from libcommon.config import ProcessingGraphConfig
from libcommon.constants import PROCESSING_STEP_CONFIG_NAMES_VERSION
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response

from .utils import (
    CONFIG_NAMES,
    CONFIG_NAMES_CONTENT,
    REVISION_NAME,
    assert_dataset_state,
    get_dataset_state,
)

PROCESSING_GRAPH = ProcessingGraph(processing_graph_specification=ProcessingGraphConfig().specification)


@pytest.fixture(autouse=True)
def queue_mongo_resource_autouse(queue_mongo_resource: QueueMongoResource) -> QueueMongoResource:
    return queue_mongo_resource


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(cache_mongo_resource: CacheMongoResource) -> CacheMongoResource:
    return cache_mongo_resource


def test_plan_job_creation_and_termination() -> None:
    # we launch all the backfill tasks
    dataset_state = get_dataset_state(processing_graph=PROCESSING_GRAPH)
    assert_dataset_state(
        dataset_state=dataset_state,
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
                "/config-names,dataset,revision",
                "dataset-info,dataset,revision",
                "dataset-is-valid,dataset,revision",
                "dataset-opt-in-out-urls-count,dataset,revision",
                "dataset-parquet,dataset,revision",
                "dataset-size,dataset,revision",
                "dataset-split-names,dataset,revision",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [],
        },
        # The queue is empty, so no step is in process.
        queue_status={"in_process": []},
        # The root dataset-level steps, as well as the "fan-in" steps, are ready to be backfilled.
        tasks=[
            "CreateJob,/config-names,dataset,revision",
            "CreateJob,dataset-info,dataset,revision",
            "CreateJob,dataset-is-valid,dataset,revision",
            "CreateJob,dataset-opt-in-out-urls-count,dataset,revision",
            "CreateJob,dataset-parquet,dataset,revision",
            "CreateJob,dataset-size,dataset,revision",
            "CreateJob,dataset-split-names,dataset,revision",
        ],
    )

    dataset_state.backfill()

    dataset_state = get_dataset_state(processing_graph=PROCESSING_GRAPH)
    assert_dataset_state(
        dataset_state=dataset_state,
        # The config names are not yet known
        config_names=[],
        # The split names are not yet known
        split_names_in_first_config=[],
        # the cache has not changed
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/config-names,dataset,revision",
                "dataset-info,dataset,revision",
                "dataset-is-valid,dataset,revision",
                "dataset-opt-in-out-urls-count,dataset,revision",
                "dataset-parquet,dataset,revision",
                "dataset-size,dataset,revision",
                "dataset-split-names,dataset,revision",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [],
        },
        # the jobs have been created and are in process
        queue_status={
            "in_process": [
                "/config-names,dataset,revision",
                "dataset-info,dataset,revision",
                "dataset-is-valid,dataset,revision",
                "dataset-opt-in-out-urls-count,dataset,revision",
                "dataset-parquet,dataset,revision",
                "dataset-size,dataset,revision",
                "dataset-split-names,dataset,revision",
            ]
        },
        # thus: no new task
        tasks=[],
    )

    # we simulate the job for "/config-names,dataset,revision" has finished
    job_info = Queue().start_job(job_types_only=["/config-names"])
    upsert_response(
        kind=job_info["type"],
        dataset=job_info["params"]["dataset"],
        config=job_info["params"]["config"],
        split=job_info["params"]["split"],
        content=CONFIG_NAMES_CONTENT,
        http_status=HTTPStatus.OK,
        job_runner_version=PROCESSING_STEP_CONFIG_NAMES_VERSION,
        dataset_git_revision=REVISION_NAME,
    )
    Queue().finish_job(job_id=job_info["job_id"], is_success=True)

    dataset_state = get_dataset_state(processing_graph=PROCESSING_GRAPH)
    assert_dataset_state(
        dataset_state=dataset_state,
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
                "/split-names-from-dataset-info,dataset,revision,config1",
                "/split-names-from-dataset-info,dataset,revision,config2",
                "config-split-names-from-streaming,dataset,revision,config1",
                "config-split-names-from-streaming,dataset,revision,config2",
                "config-info,dataset,revision,config1",
                "config-info,dataset,revision,config2",
                "config-opt-in-out-urls-count,dataset,revision,config1",
                "config-opt-in-out-urls-count,dataset,revision,config2",
                "config-parquet,dataset,revision,config1",
                "config-parquet,dataset,revision,config2",
                "config-parquet-and-info,dataset,revision,config1",
                "config-parquet-and-info,dataset,revision,config2",
                "config-parquet-metadata,dataset,revision,config1",
                "config-parquet-metadata,dataset,revision,config2",
                "config-size,dataset,revision,config1",
                "config-size,dataset,revision,config2",
                "dataset-info,dataset,revision",
                "dataset-is-valid,dataset,revision",
                "dataset-opt-in-out-urls-count,dataset,revision",
                "dataset-parquet,dataset,revision",
                "dataset-size,dataset,revision",
                "dataset-split-names,dataset,revision",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": ["/config-names,dataset,revision"],
        },
        # the job "/config-names,dataset,revision" is no more in process
        queue_status={
            "in_process": [
                "dataset-info,dataset,revision",
                "dataset-is-valid,dataset,revision",
                "dataset-opt-in-out-urls-count,dataset,revision",
                "dataset-parquet,dataset,revision",
                "dataset-size,dataset,revision",
                "dataset-split-names,dataset,revision",
            ]
        },
        tasks=[
            "CreateJob,/split-names-from-dataset-info,dataset,revision,config1",
            "CreateJob,/split-names-from-dataset-info,dataset,revision,config2",
            "CreateJob,config-split-names-from-streaming,dataset,revision,config1",
            "CreateJob,config-split-names-from-streaming,dataset,revision,config2",
            "CreateJob,config-info,dataset,revision,config1",
            "CreateJob,config-info,dataset,revision,config2",
            "CreateJob,config-opt-in-out-urls-count,dataset,revision,config1",
            "CreateJob,config-opt-in-out-urls-count,dataset,revision,config2",
            "CreateJob,config-parquet,dataset,revision,config1",
            "CreateJob,config-parquet,dataset,revision,config2",
            "CreateJob,config-parquet-and-info,dataset,revision,config1",
            "CreateJob,config-parquet-and-info,dataset,revision,config2",
            "CreateJob,config-parquet-metadata,dataset,revision,config1",
            "CreateJob,config-parquet-metadata,dataset,revision,config2",
            "CreateJob,config-size,dataset,revision,config1",
            "CreateJob,config-size,dataset,revision,config2",
        ],
    )
