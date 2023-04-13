# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Iterator, List, Mapping, Optional, TypedDict

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
    CacheState,
    ConfigState,
    DatasetState,
    JobState,
    SplitState,
    StepState,
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


def test_step_state_as_dict() -> None:
    dataset = DATASET_NAME
    config = None
    split = None
    step = PROCESSING_GRAPH.get_step(name="/config-names")
    step_state = StepState(dataset=dataset, config=config, split=split, step=step)
    assert step_state.as_dict() == {
        "step_name": "/config-names",
        "job_state": {"is_in_process": False},
        "cache_state": {"exists": False, "is_success": False},
        "should_be_backfilled": True,
    }
    assert step_state.id == f"/config-names[{dataset},{config},{split}]"


def test_step_state_backfill() -> None:
    dataset = DATASET_NAME
    config = None
    split = None
    step = PROCESSING_GRAPH.get_step(name="/config-names")
    step_state = StepState(dataset=dataset, config=config, split=split, step=step)
    assert not step_state.cache_state.exists
    assert not step_state.job_state.is_in_process
    assert step_state.should_be_backfilled()
    step_state.backfill()
    step_state = StepState(dataset=dataset, config=config, split=split, step=step)
    assert not step_state.cache_state.exists
    assert step_state.job_state.is_in_process
    assert not step_state.should_be_backfilled()


SPLIT1_NAME = "split1"
SPLIT1_STATE_DICT = {
    "split": SPLIT1_NAME,
    "step_states": [
        {
            "step_name": "split-first-rows-from-streaming",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
        },
        {
            "step_name": "split-first-rows-from-parquet",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
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
    "step_states": [
        {
            "step_name": "split-first-rows-from-streaming",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
        },
        {
            "step_name": "split-first-rows-from-parquet",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
        },
    ],
}

CONFIG_STATE_DICT = {
    "config": "config",
    "split_states": [
        SPLIT1_STATE_DICT,
        SPLIT2_STATE_DICT,
    ],
    "step_states": [
        {
            "step_name": "/split-names-from-streaming",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
        },
        {
            "step_name": "config-parquet-and-info",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
        },
        {
            "step_name": "config-parquet",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
        },
        {
            "step_name": "config-info",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
        },
        {
            "step_name": "/split-names-from-dataset-info",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": True, "is_success": True},  # <- this entry is in the cache
            "should_be_backfilled": False,  # <- thus: no need to backfill
        },
        {
            "step_name": "config-size",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
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
        "step_states": [
            {
                "step_name": "/config-names",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": True, "is_success": True},  # <- this entry is in the cache
                "should_be_backfilled": False,  # <- thus: no need to backfill
            },
            {
                "step_name": "/parquet-and-dataset-info",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
                "should_be_backfilled": True,
            },
            {
                "step_name": "dataset-parquet",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
                "should_be_backfilled": True,
            },
            {
                "step_name": "dataset-info",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
                "should_be_backfilled": True,
            },
            {
                "step_name": "dataset-size",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
                "should_be_backfilled": True,
            },
            {
                "step_name": "dataset-split-names-from-streaming",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
                "should_be_backfilled": True,
            },
            {
                "step_name": "dataset-split-names-from-dataset-info",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
                "should_be_backfilled": True,
            },
            {
                "step_name": "dataset-split-names",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
                "should_be_backfilled": True,
            },
            {
                "step_name": "dataset-is-valid",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
                "should_be_backfilled": True,
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


def test_get_backfill_tasks() -> None:  # sourcery skip: extract-duplicate-method
    dataset = DATASET_NAME
    processing_graph = PROCESSING_GRAPH
    dataset_state = DatasetState(dataset=dataset, processing_graph=processing_graph)
    assert not dataset_state.config_names
    # Note that no config-level and split-level step is listed here, because the config names and splits names are not
    # yet known.
    # The root dataset-level steps are ready to be backfilled.
    assert dataset_state.step_states_by_status.get_ids() == {
        "blocked_by_parent": [],
        "should_be_backfilled": ["/config-names[dataset,None,None]", "/parquet-and-dataset-info[dataset,None,None]"],
        "in_process": [],
        "up_to_date": [],
        "undefined": [
            "dataset-info[dataset,None,None]",
            "dataset-is-valid[dataset,None,None]",
            "dataset-parquet[dataset,None,None]",
            "dataset-size[dataset,None,None]",
            "dataset-split-names-from-dataset-info[dataset,None,None]",
            "dataset-split-names-from-streaming[dataset,None,None]",
            "dataset-split-names[dataset,None,None]",
        ],
    }

    # we launch all the backfill tasks
    dataset_state.backfill()

    # the jobs have been created and are in process, and the cache has not changed
    # thus: no new backfill task is proposed, and the state steps are now in "in_process" status
    dataset_state = DatasetState(dataset=dataset, processing_graph=processing_graph)
    assert not dataset_state.config_names
    assert dataset_state.step_states_by_status.get_ids() == {
        "blocked_by_parent": [],
        "should_be_backfilled": [],
        "in_process": ["/config-names[dataset,None,None]", "/parquet-and-dataset-info[dataset,None,None]"],
        "up_to_date": [],
        "undefined": [
            "dataset-info[dataset,None,None]",
            "dataset-is-valid[dataset,None,None]",
            "dataset-parquet[dataset,None,None]",
            "dataset-size[dataset,None,None]",
            "dataset-split-names-from-dataset-info[dataset,None,None]",
            "dataset-split-names-from-streaming[dataset,None,None]",
            "dataset-split-names[dataset,None,None]",
        ],
    }

    # simulate that the "backfill[/config-names,dataset,None,None]" task has finished
    finish_task(job_type="/config-names", content=ONE_CONFIG_NAME_CONTENT_OK)

    # The "/config-names" step is now up-to-date
    # The first config-level steps are ready to be backfilled for all the configs (only one in this case)
    # New step states also appear in the "blocked_by_parent" status, because the config names are now known, but the
    # parents are not ready yet.
    # The split-level steps are still missing, because the splits names are not yet known, for any config.
    dataset_state = DatasetState(dataset=dataset, processing_graph=processing_graph)
    assert dataset_state.config_names == [CONFIG_NAME]
    assert len(dataset_state.config_states) == 1
    assert dataset_state.config_states[0].split_names == []
    assert dataset_state.step_states_by_status.get_ids() == {
        "blocked_by_parent": [
            "/split-names-from-dataset-info[dataset,config,None]",
            "config-info[dataset,config,None]",
            "config-parquet[dataset,config,None]",
            "config-size[dataset,config,None]",
        ],
        "should_be_backfilled": [
            "/split-names-from-streaming[dataset,config,None]",
            "config-parquet-and-info[dataset,config,None]",
        ],
        "in_process": ["/parquet-and-dataset-info[dataset,None,None]"],
        "up_to_date": ["/config-names[dataset,None,None]"],
        "undefined": [
            "dataset-info[dataset,None,None]",
            "dataset-is-valid[dataset,None,None]",
            "dataset-parquet[dataset,None,None]",
            "dataset-size[dataset,None,None]",
            "dataset-split-names-from-dataset-info[dataset,None,None]",
            "dataset-split-names-from-streaming[dataset,None,None]",
            "dataset-split-names[dataset,None,None]",
        ],
    }

    # launch the backfill tasks
    dataset_state.backfill()
    # and simulate that the "backfill[config-parquet-and-info,dataset,config,None]" task has finished
    finish_task(job_type="config-parquet-and-info", content=CONFIG_PARQUET_AND_INFO_OK)

    # the config-level dependent steps are now ready to be backfilled
    # (they are no more blocked by "config-parquet-and-info", which is up-to-date)
    dataset_state = DatasetState(dataset=dataset, processing_graph=processing_graph)
    assert dataset_state.config_names == [CONFIG_NAME]
    assert len(dataset_state.config_states) == 1
    assert dataset_state.config_states[0].split_names == []
    assert dataset_state.step_states_by_status.get_ids() == {
        "blocked_by_parent": ["/split-names-from-dataset-info[dataset,config,None]"],
        "should_be_backfilled": [
            "config-info[dataset,config,None]",
            "config-parquet[dataset,config,None]",
            "config-size[dataset,config,None]",
        ],
        "in_process": [
            "/parquet-and-dataset-info[dataset,None,None]",
            "/split-names-from-streaming[dataset,config,None]",
        ],
        "up_to_date": ["/config-names[dataset,None,None]", "config-parquet-and-info[dataset,config,None]"],
        "undefined": [
            "dataset-info[dataset,None,None]",
            "dataset-is-valid[dataset,None,None]",
            "dataset-parquet[dataset,None,None]",
            "dataset-size[dataset,None,None]",
            "dataset-split-names-from-dataset-info[dataset,None,None]",
            "dataset-split-names-from-streaming[dataset,None,None]",
            "dataset-split-names[dataset,None,None]",
        ],
    }

    # launch the backfill tasks
    dataset_state.backfill()
    # and simulate that the "backfill[config-info,dataset,config,None]" task has finished
    finish_task(job_type="config-info", content=CONFIG_INFO_OK)

    # "config-info" is up-to-date
    # "/split-names-from-dataset-info" is no more blocked, and is ready to be backfilled
    dataset_state = DatasetState(dataset=dataset, processing_graph=processing_graph)
    assert dataset_state.config_names == [CONFIG_NAME]
    assert len(dataset_state.config_states) == 1
    assert dataset_state.config_states[0].split_names == []
    assert dataset_state.step_states_by_status.get_ids() == {
        "blocked_by_parent": [],
        "should_be_backfilled": ["/split-names-from-dataset-info[dataset,config,None]"],
        "in_process": [
            "/parquet-and-dataset-info[dataset,None,None]",
            "/split-names-from-streaming[dataset,config,None]",
            "config-parquet[dataset,config,None]",
            "config-size[dataset,config,None]",
        ],
        "up_to_date": [
            "/config-names[dataset,None,None]",
            "config-info[dataset,config,None]",
            "config-parquet-and-info[dataset,config,None]",
        ],
        "undefined": [
            "dataset-info[dataset,None,None]",
            "dataset-is-valid[dataset,None,None]",
            "dataset-parquet[dataset,None,None]",
            "dataset-size[dataset,None,None]",
            "dataset-split-names-from-dataset-info[dataset,None,None]",
            "dataset-split-names-from-streaming[dataset,None,None]",
            "dataset-split-names[dataset,None,None]",
        ],
    }

    # launch the backfill tasks
    dataset_state.backfill()
    # simulate that the "backfill[/split-names-from-dataset-info,dataset,config,None]" task and
    # the "backfill[/split-names-from-streaming,dataset,config,None]" have finished
    finish_task(job_type="/split-names-from-dataset-info", content=SPLIT_NAMES_RESPONSE_OK["content"])
    finish_task(job_type="/split-names-from-streaming", content=SPLIT_NAMES_RESPONSE_OK["content"])

    # "/split-names-from-dataset-info" and "/split-names-from-streaming" are up-to-date for the config
    # the split names are now available
    # the split-level dependent steps are now ready to be backfilled, or blocked by parent
    dataset_state = DatasetState(dataset=dataset, processing_graph=processing_graph)
    assert dataset_state.config_names == [CONFIG_NAME]
    assert len(dataset_state.config_states) == 1
    assert dataset_state.config_states[0].split_names == SPLIT_NAMES_OK
    # the split-level dependent steps are now ready to be backfilled
    assert dataset_state.step_states_by_status.get_ids() == {
        "blocked_by_parent": [
            "split-first-rows-from-parquet[dataset,config,split1]",
            "split-first-rows-from-parquet[dataset,config,split2]",
        ],
        "should_be_backfilled": [
            "split-first-rows-from-streaming[dataset,config,split1]",
            "split-first-rows-from-streaming[dataset,config,split2]",
        ],
        "in_process": [
            "/parquet-and-dataset-info[dataset,None,None]",
            "config-parquet[dataset,config,None]",
            "config-size[dataset,config,None]",
        ],
        "up_to_date": [
            "/config-names[dataset,None,None]",
            "/split-names-from-dataset-info[dataset,config,None]",
            "/split-names-from-streaming[dataset,config,None]",
            "config-info[dataset,config,None]",
            "config-parquet-and-info[dataset,config,None]",
        ],
        "undefined": [
            "dataset-info[dataset,None,None]",
            "dataset-is-valid[dataset,None,None]",
            "dataset-parquet[dataset,None,None]",
            "dataset-size[dataset,None,None]",
            "dataset-split-names-from-dataset-info[dataset,None,None]",
            "dataset-split-names-from-streaming[dataset,None,None]",
            "dataset-split-names[dataset,None,None]",
        ],
    }
