# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Dict, Iterator, List, Mapping, Optional, TypedDict

import pytest

from libcommon.config import ProcessingGraphConfig
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue, Status, _clean_queue_database
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import _clean_cache_database, upsert_response
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
def queue_mongo_resource(queue_mongo_host: str) -> Iterator[QueueMongoResource]:
    database = "datasets_server_queue_test"
    host = queue_mongo_host
    if "test" not in database:
        raise ValueError("Test must be launched on a test mongo database")
    with QueueMongoResource(database=database, host=host, server_selection_timeout_ms=3_000) as queue_mongo_resource:
        if not queue_mongo_resource.is_available():
            raise RuntimeError("Mongo resource is not available")
        yield queue_mongo_resource
        _clean_queue_database()


@pytest.fixture(autouse=True)
def cache_mongo_resource(cache_mongo_host: str) -> Iterator[CacheMongoResource]:
    database = "datasets_server_cache_test"
    host = cache_mongo_host
    if "test" not in database:
        raise ValueError("Test must be launched on a test mongo database")
    with CacheMongoResource(database=database, host=host) as cache_mongo_resource:
        yield cache_mongo_resource
        _clean_cache_database()


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


CONFIG_NAME = "config"
SPLIT_NAMES_OK = ["split1", "split2"]
SPLIT_NAMES_RESPONSE_OK = ResponseSpec(
    content={
        "split_names": [
            {"dataset": DATASET_NAME, "config": CONFIG_NAME, "split": split_name} for split_name in SPLIT_NAMES_OK
        ]
    },
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
            config=CONFIG_NAME,
            split=None,
            content=response_spec["content"],
            http_status=response_spec["http_status"],
        )

    if raises:
        with pytest.raises(Exception):
            fetch_split_names(dataset=DATASET_NAME, config=CONFIG_NAME)
    else:
        split_names = fetch_split_names(dataset=DATASET_NAME, config=CONFIG_NAME)
        assert split_names == expected_split_names


SPLIT_NAME = "split"
JOB_TYPE = "job_type"


@pytest.mark.parametrize(
    "dataset,config,split,job_type",
    [
        (DATASET_NAME, None, None, JOB_TYPE),
        (DATASET_NAME, CONFIG_NAME, None, JOB_TYPE),
        (DATASET_NAME, CONFIG_NAME, SPLIT_NAME, JOB_TYPE),
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
        (DATASET_NAME, CONFIG_NAME, None, JOB_TYPE),
        (DATASET_NAME, CONFIG_NAME, SPLIT_NAME, JOB_TYPE),
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
        (DATASET_NAME, CONFIG_NAME, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME, SPLIT_NAME, CACHE_KIND),
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
        (DATASET_NAME, CONFIG_NAME, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME, SPLIT_NAME, CACHE_KIND),
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
        (DATASET_NAME, CONFIG_NAME, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME, SPLIT_NAME, CACHE_KIND),
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


SPLIT1_NAME = "split1"
SPLIT1_STATE_DICT = {
    "split": SPLIT1_NAME,
    "artifact_states": [
        {
            "id": f"split-first-rows-from-streaming,{DATASET_NAME},{CONFIG_NAME},{SPLIT1_NAME}",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
        },
        {
            "id": f"split-first-rows-from-parquet,{DATASET_NAME},{CONFIG_NAME},{SPLIT1_NAME}",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
        },
    ],
}


def test_split_state_as_dict() -> None:
    dataset = DATASET_NAME
    config = CONFIG_NAME
    split = SPLIT1_NAME
    processing_graph = PROCESSING_GRAPH
    assert (
        SplitState(dataset=dataset, config=config, split=split, processing_graph=processing_graph).as_dict()
        == SPLIT1_STATE_DICT
    )


SPLIT2_NAME = "split2"
SPLIT2_STATE_DICT = {
    "split": SPLIT2_NAME,
    "artifact_states": [
        {
            "id": f"split-first-rows-from-streaming,{DATASET_NAME},{CONFIG_NAME},{SPLIT2_NAME}",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
        },
        {
            "id": f"split-first-rows-from-parquet,{DATASET_NAME},{CONFIG_NAME},{SPLIT2_NAME}",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
        },
    ],
}

CONFIG_STATE_DICT = {
    "config": "config",
    "split_states": [
        SPLIT1_STATE_DICT,
        SPLIT2_STATE_DICT,
    ],
    "artifact_states": [
        {
            "id": f"/split-names-from-streaming,{DATASET_NAME},{CONFIG_NAME}",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
        },
        {
            "id": f"config-parquet-and-info,{DATASET_NAME},{CONFIG_NAME}",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
        },
        {
            "id": f"config-parquet,{DATASET_NAME},{CONFIG_NAME}",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
        },
        {
            "id": f"config-info,{DATASET_NAME},{CONFIG_NAME}",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
        },
        {
            "id": f"/split-names-from-dataset-info,{DATASET_NAME},{CONFIG_NAME}",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": True, "is_success": True},  # <- this entry is in the cache
        },
        {
            "id": f"config-size,{DATASET_NAME},{CONFIG_NAME}",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
        },
    ],
}


def test_config_state_as_dict() -> None:
    dataset = DATASET_NAME
    config = CONFIG_NAME
    upsert_response(
        kind=HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND,
        dataset=DATASET_NAME,
        config=CONFIG_NAME,
        split=None,
        content=SPLIT_NAMES_RESPONSE_OK["content"],
        http_status=SPLIT_NAMES_RESPONSE_OK["http_status"],
    )
    processing_graph = PROCESSING_GRAPH
    assert (
        ConfigState(dataset=dataset, config=config, processing_graph=processing_graph).as_dict() == CONFIG_STATE_DICT
    )


ONE_CONFIG_NAME_CONTENT_OK = {"config_names": [{"config": CONFIG_NAME}]}


def test_dataset_state_as_dict() -> None:
    dataset = DATASET_NAME
    upsert_response(
        kind=HARD_CODED_CONFIG_NAMES_CACHE_KIND,
        dataset=DATASET_NAME,
        config=None,
        split=None,
        content=ONE_CONFIG_NAME_CONTENT_OK,
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind=HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND,
        dataset=DATASET_NAME,
        config=CONFIG_NAME,
        split=None,
        content=SPLIT_NAMES_RESPONSE_OK["content"],
        http_status=SPLIT_NAMES_RESPONSE_OK["http_status"],
    )
    processing_graph = PROCESSING_GRAPH
    assert DatasetState(dataset=dataset, processing_graph=processing_graph).as_dict() == {
        "dataset": "dataset",
        "config_states": [CONFIG_STATE_DICT],
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


CONFIG_PARQUET_AND_INFO_OK = {"config": CONFIG_NAME, "content": "not important"}
CONFIG_INFO_OK = {"config": CONFIG_NAME, "content": "not important"}


def finish_task(job_type: str, content: Any) -> None:
    job_info = Queue().start_job(only_job_types=[job_type])
    upsert_response(
        kind=job_info["type"],
        dataset=job_info["dataset"],
        config=job_info["config"],
        split=job_info["split"],
        content=content,
        http_status=HTTPStatus.OK,
    )
    Queue().finish_job(job_id=job_info["job_id"], finished_status=Status.SUCCESS)


def test_backfill() -> None:
    dataset = DATASET_NAME
    processing_graph = PROCESSING_GRAPH

    def assert_dataset_state(
        config_names: List[str],
        split_names_in_first_config: List[str],
        cache_status: Dict[str, List[str]],
        queue_status: Dict[str, List[str]],
        tasks: List[str],
    ) -> DatasetState:
        dataset_state = DatasetState(dataset=dataset, processing_graph=processing_graph)
        assert dataset_state.config_names == config_names
        assert len(dataset_state.config_states) == len(config_names)
        if len(config_names):
            assert dataset_state.config_states[0].split_names == split_names_in_first_config
        else:
            # this case is just to check the test, not the code
            assert not split_names_in_first_config
        assert {
            "blocked_by_parent": sorted(dataset_state.cache_status.blocked_by_parent.keys()),
            "cache_is_outdated_by_parent": sorted(dataset_state.cache_status.cache_is_outdated_by_parent.keys()),
            "cache_is_empty": sorted(dataset_state.cache_status.cache_is_empty.keys()),
            "cache_is_error_to_retry": sorted(dataset_state.cache_status.cache_is_error_to_retry.keys()),
            "up_to_date": sorted(dataset_state.cache_status.up_to_date.keys()),
        } == cache_status
        assert {"in_process": sorted(dataset_state.queue_status.in_process.keys())} == queue_status
        assert sorted(task.id for task in dataset_state.plan.tasks) == tasks
        return dataset_state

    dataset_state = assert_dataset_state(
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

    # we launch all the backfill tasks
    dataset_state.backfill()

    assert_dataset_state(
        # The config names are not yet known
        config_names=[],
        # The split names are not yet known
        split_names_in_first_config=[],
        # the cache has not changed
        cache_status={
            "blocked_by_parent": ["dataset-is-valid,dataset"],
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

    # simulate that the "backfill[/config-names,dataset]" task has finished
    finish_task(job_type="/config-names", content=ONE_CONFIG_NAME_CONTENT_OK)

    dataset_state = assert_dataset_state(
        # The config names are known
        config_names=[CONFIG_NAME],
        # The split names are not yet known
        split_names_in_first_config=[],
        # The "/config-names" step is now up-to-date
        # New step states appear in the "blocked_by_parent" status, because the config names are now known, but the
        # parents are not ready yet.
        # The split-level steps are still missing, because the splits names are not yet known, for any config.
        cache_status={
            "blocked_by_parent": [
                "/split-names-from-dataset-info,dataset,config",
                "config-info,dataset,config",
                "config-parquet,dataset,config",
                "config-size,dataset,config",
                "dataset-is-valid,dataset",
            ],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/parquet-and-dataset-info,dataset",
                "/split-names-from-streaming,dataset,config",
                "config-parquet-and-info,dataset,config",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
            ],
            "cache_is_error_to_retry": [],
            "up_to_date": ["/config-names,dataset"],
        },
        # the "/config-names" job has finished
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
        # The next config-level steps in the graph are ready to be backfilled for each config
        # (only one config in our test)
        tasks=[
            "CreateJob[/split-names-from-streaming,dataset,config]",
            "CreateJob[config-parquet-and-info,dataset,config]",
        ],
    )

    # launch the backfill tasks
    dataset_state.backfill()
    # and simulate that the "backfill[config-parquet-and-info,dataset,config]" task has finished
    finish_task(job_type="config-parquet-and-info", content=CONFIG_PARQUET_AND_INFO_OK)

    dataset_state = assert_dataset_state(
        # The config names are known
        config_names=[CONFIG_NAME],
        # The split names are not yet known
        split_names_in_first_config=[],
        # the config-level dependent steps are no more blocked by "config-parquet-and-info", which is up-to-date
        cache_status={
            "blocked_by_parent": [
                "/split-names-from-dataset-info,dataset,config",
                "dataset-is-valid,dataset",
            ],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/parquet-and-dataset-info,dataset",
                "/split-names-from-streaming,dataset,config",
                "config-info,dataset,config",
                "config-parquet,dataset,config",
                "config-size,dataset,config",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
            ],
            "cache_is_error_to_retry": [],
            "up_to_date": ["/config-names,dataset", "config-parquet-and-info,dataset,config"],
        },
        # The "config-parquet-and-info" job has finished, while the other config-level step is still in progress
        queue_status={
            "in_process": [
                "/parquet-and-dataset-info,dataset",
                "/split-names-from-streaming,dataset,config",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
            ]
        },
        # the config-level dependent steps are now ready to be backfilled
        tasks=[
            "CreateJob[config-info,dataset,config]",
            "CreateJob[config-parquet,dataset,config]",
            "CreateJob[config-size,dataset,config]",
        ],
    )

    # launch the backfill tasks
    dataset_state.backfill()
    # and simulate that the "backfill[config-info,dataset,config]" task has finished
    finish_task(job_type="config-info", content=CONFIG_INFO_OK)

    # "config-info" is up-to-date
    # "/split-names-from-dataset-info" is no more blocked, and is ready to be backfilled
    dataset_state = assert_dataset_state(
        # The config names are known
        config_names=[CONFIG_NAME],
        # The split names are not yet known
        split_names_in_first_config=[],
        cache_status={
            "blocked_by_parent": ["dataset-is-valid,dataset"],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/parquet-and-dataset-info,dataset",
                "/split-names-from-dataset-info,dataset,config",
                "/split-names-from-streaming,dataset,config",
                "config-parquet,dataset,config",
                "config-size,dataset,config",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
            ],
            "cache_is_error_to_retry": [],
            "up_to_date": [
                "/config-names,dataset",
                "config-info,dataset,config",
                "config-parquet-and-info,dataset,config",
            ],
        },
        queue_status={
            "in_process": [
                "/parquet-and-dataset-info,dataset",
                "/split-names-from-streaming,dataset,config",
                "config-parquet,dataset,config",
                "config-size,dataset,config",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
            ]
        },
        tasks=["CreateJob[/split-names-from-dataset-info,dataset,config]"],
    )

    # launch the backfill tasks
    dataset_state.backfill()
    # simulate that the "backfill[/split-names-from-dataset-info,dataset,config]" task and
    # the "backfill[/split-names-from-streaming,dataset,config]" have finished
    finish_task(job_type="/split-names-from-dataset-info", content=SPLIT_NAMES_RESPONSE_OK["content"])
    finish_task(job_type="/split-names-from-streaming", content=SPLIT_NAMES_RESPONSE_OK["content"])

    # "/split-names-from-dataset-info" and "/split-names-from-streaming" are up-to-date for the config
    # the split names are now available
    # the split-level dependent steps are now ready to be backfilled, or blocked by parent
    dataset_state = assert_dataset_state(
        # The config names are known
        config_names=[CONFIG_NAME],
        # The split names are known
        split_names_in_first_config=SPLIT_NAMES_OK,
        cache_status={
            "blocked_by_parent": [
                "dataset-is-valid,dataset",
                "split-first-rows-from-parquet,dataset,config,split1",
                "split-first-rows-from-parquet,dataset,config,split2",
            ],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "/parquet-and-dataset-info,dataset",
                "config-parquet,dataset,config",
                "config-size,dataset,config",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
                "split-first-rows-from-streaming,dataset,config,split1",
                "split-first-rows-from-streaming,dataset,config,split2",
            ],
            "cache_is_error_to_retry": [],
            "up_to_date": [
                "/config-names,dataset",
                "/split-names-from-dataset-info,dataset,config",
                "/split-names-from-streaming,dataset,config",
                "config-info,dataset,config",
                "config-parquet-and-info,dataset,config",
            ],
        },
        queue_status={
            "in_process": [
                "/parquet-and-dataset-info,dataset",
                "config-parquet,dataset,config",
                "config-size,dataset,config",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
            ]
        },
        tasks=[
            "CreateJob[split-first-rows-from-streaming,dataset,config,split1]",
            "CreateJob[split-first-rows-from-streaming,dataset,config,split2]",
        ],
    )

    # force an update of the /config-names step
    upsert_response(
        kind="/config-names",
        dataset="dataset",
        content=ONE_CONFIG_NAME_CONTENT_OK,
        http_status=HTTPStatus.OK,
    )

    # As the date of the /config-names step is updated, all the steps that depend on it are
    # considered as outdated, and are now in the "cache_is_outdated_by_parent" status, for
    # the first children steps. The following descendants steps are in "blocked_by_parent"
    # status, even those that were already in in_process status -> their jobs will be deleted.
    # the split-level dependent steps are now ready to be backfilled
    dataset_state = assert_dataset_state(
        # The config names are known
        config_names=[CONFIG_NAME],
        # The split names are known
        split_names_in_first_config=SPLIT_NAMES_OK,
        cache_status={
            "blocked_by_parent": [
                "/split-names-from-dataset-info,dataset,config",
                "config-info,dataset,config",
                "config-parquet,dataset,config",
                "config-size,dataset,config",
                "dataset-is-valid,dataset",
                "split-first-rows-from-parquet,dataset,config,split1",
                "split-first-rows-from-parquet,dataset,config,split2",
                "split-first-rows-from-streaming,dataset,config,split1",
                "split-first-rows-from-streaming,dataset,config,split2",
            ],
            "cache_is_outdated_by_parent": [
                "/split-names-from-streaming,dataset,config",
                "config-parquet-and-info,dataset,config",
            ],
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
            "up_to_date": ["/config-names,dataset"],
        },
        queue_status={
            "in_process": [
                "/parquet-and-dataset-info,dataset",
                "config-parquet,dataset,config",
                "config-size,dataset,config",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
            ]
        },
        tasks=[
            "CreateJob[/split-names-from-streaming,dataset,config]",
            "CreateJob[config-parquet-and-info,dataset,config]",
            "DeleteJob[config-parquet,dataset,config]",
            "DeleteJob[config-size,dataset,config]",
        ],
    )

    # launch the backfill tasks
    dataset_state.backfill()

    # the split-level dependent steps are now ready to be backfilled
    dataset_state = assert_dataset_state(
        # The config names are known
        config_names=[CONFIG_NAME],
        # The split names are known
        split_names_in_first_config=SPLIT_NAMES_OK,
        # the cache content has not changed
        cache_status={
            "blocked_by_parent": [
                "/split-names-from-dataset-info,dataset,config",
                "config-info,dataset,config",
                "config-parquet,dataset,config",
                "config-size,dataset,config",
                "dataset-is-valid,dataset",
                "split-first-rows-from-parquet,dataset,config,split1",
                "split-first-rows-from-parquet,dataset,config,split2",
                "split-first-rows-from-streaming,dataset,config,split1",
                "split-first-rows-from-streaming,dataset,config,split2",
            ],
            "cache_is_outdated_by_parent": [
                "/split-names-from-streaming,dataset,config",
                "config-parquet-and-info,dataset,config",
            ],
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
            "up_to_date": ["/config-names,dataset"],
        },
        # the jobs of blocked step states have been deleted, thus they are no more listed as in process
        queue_status={
            "in_process": [
                "/parquet-and-dataset-info,dataset",
                "/split-names-from-streaming,dataset,config",
                "config-parquet-and-info,dataset,config",
                "dataset-info,dataset",
                "dataset-parquet,dataset",
                "dataset-size,dataset",
                "dataset-split-names,dataset",
                "dataset-split-names-from-dataset-info,dataset",
                "dataset-split-names-from-streaming,dataset",
            ]
        },
        # no new task
        tasks=[],
    )
