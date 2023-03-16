# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus

from libcommon.config import ProcessingGraphConfig
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue
from libcommon.simple_cache import upsert_response
from pytest import raises

from api.config import AppConfig, EndpointConfig
from api.routes.endpoint import EndpointsDefinition, get_cache_entry_from_steps
from api.utils import ResponseNotReadyError


def test_endpoints_definition() -> None:
    config = ProcessingGraphConfig()
    graph = ProcessingGraph(config.specification)
    endpoint_config = EndpointConfig.from_env()

    endpoints_definition = EndpointsDefinition(graph, endpoint_config)
    assert endpoints_definition

    definition = endpoints_definition.steps_by_input_type_and_endpoint
    assert definition

    config_names = definition["/config-names"]
    assert config_names is not None
    assert config_names["dataset"] is not None
    assert len(config_names["dataset"]) == 1  # Only has one processing step

    splits = definition["/splits"]
    assert splits is not None
    assert splits["dataset"] is not None
    assert len(splits["dataset"]) == 2  # Has two processing steps
    assert len(splits["config"]) == 2  # Has two processing steps

    first_rows = definition["/first-rows"]
    assert first_rows is not None
    assert first_rows["split"] is not None
    assert len(first_rows["split"]) == 1  # Only has one processing step

    parquet_and_dataset_info = definition["/parquet-and-dataset-info"]
    assert parquet_and_dataset_info is not None
    assert parquet_and_dataset_info["dataset"] is not None
    assert len(parquet_and_dataset_info["dataset"]) == 1  # Only has one processing step

    parquet = definition["/parquet"]
    assert parquet is not None
    assert parquet["dataset"] is not None
    assert len(parquet["dataset"]) == 1  # Only has one processing step

    dataset_info = definition["/dataset-info"]
    assert dataset_info is not None
    assert dataset_info["dataset"] is not None
    assert len(dataset_info["dataset"]) == 1  # Only has one processing step

    size = definition["/size"]
    assert size is not None
    assert size["dataset"] is not None
    assert size["config"] is not None
    assert len(size["dataset"]) == 1  # Only has one processing step
    assert len(size["config"]) == 1  # Only has one processing step


def test_get_cache_entry_from_steps() -> None:
    dataset = "dataset"
    config = "config"

    app_config = AppConfig.from_env()
    graph_config = ProcessingGraphConfig()
    graph = ProcessingGraph(graph_config.specification)
    init_processing_steps = graph.get_first_steps()

    cache_with_error = "/split-names-from-streaming"
    cache_without_error = "/split-names-from-dataset-info"

    step_with_error = graph.get_step(cache_with_error)
    step_whitout_error = graph.get_step(cache_without_error)

    upsert_response(
        kind=cache_without_error,
        dataset=dataset,
        config=config,
        content={},
        http_status=HTTPStatus.OK,
    )

    upsert_response(
        kind=cache_with_error,
        dataset=dataset,
        config=config,
        content={},
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    )

    # succeeded result is returned
    result = get_cache_entry_from_steps(
        [step_whitout_error, step_with_error],
        dataset,
        config,
        None,
        init_processing_steps,
        app_config.common.hf_endpoint,
    )
    assert result
    assert result["http_status"] == HTTPStatus.OK

    # succeeded result is returned even if first step failed
    result = get_cache_entry_from_steps(
        [step_with_error, step_whitout_error],
        dataset,
        config,
        None,
        init_processing_steps,
        app_config.common.hf_endpoint,
    )
    assert result
    assert result["http_status"] == HTTPStatus.OK

    # error result is returned if all steps failed
    result = get_cache_entry_from_steps(
        [step_with_error, step_with_error], dataset, config, None, init_processing_steps, app_config.common.hf_endpoint
    )
    assert result
    assert result["http_status"] == HTTPStatus.INTERNAL_SERVER_ERROR

    # peding job thows exception
    queue = Queue()
    queue.upsert_job(job_type="/splits", dataset=dataset, config=config, force=True)
    non_existent_step = graph.get_step("/splits")
    with raises(ResponseNotReadyError):
        get_cache_entry_from_steps(
            [non_existent_step], dataset, config, None, init_processing_steps, app_config.common.hf_endpoint
        )
