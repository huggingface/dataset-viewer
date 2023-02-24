# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus

from libcommon.config import ProcessingGraphConfig
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue
from libcommon.simple_cache import CacheEntry, upsert_response
from pytest import raises
from starlette.datastructures import QueryParams

from api.config import AppConfig, EndpointConfig
from api.routes.endpoint import (
    EndpointsDefinition,
    InputParameters,
    get_first_succeeded_cache_entry_from_steps,
    get_input_parameters,
    get_response_from_cache_entry,
)
from api.utils import MissingRequiredParameterError, ResponseNotReadyError


def test_endpoints_definition() -> None:
    config = ProcessingGraphConfig()
    graph = ProcessingGraph(config.specification)
    endpoint_config = EndpointConfig.from_env()

    endpoints_definition = EndpointsDefinition(graph, endpoint_config)
    assert endpoints_definition

    definition = endpoints_definition.processing_steps_by_endpoint
    assert definition

    config_names = definition["/config-names"]
    assert config_names is not None
    assert config_names["dataset"] is not None
    assert len(config_names["dataset"]) == 1  # Only has one processing step

    split_names_from_streaming = definition["/split-names-from-streaming"]
    assert split_names_from_streaming is not None
    assert split_names_from_streaming["config"] is not None
    assert len(split_names_from_streaming["config"]) == 1  # Only has one processing step

    splits = definition["/splits"]
    assert splits is not None
    assert splits["dataset"] is not None
    assert len(splits["dataset"]) == 1  # Only has one processing step
    assert len(splits["config"]) == 2  # Has two processing step

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

    sizes = definition["/sizes"]
    assert sizes is not None
    assert sizes["dataset"] is not None
    assert len(sizes["dataset"]) == 1  # Only has one processing step


def test_get_input_parameters() -> None:
    query_params = QueryParams([("config", None)])
    with raises(MissingRequiredParameterError) as e:
        get_input_parameters(query_params=query_params)
    assert e.value.message == "Parameter 'dataset' is required"

    query_params = QueryParams([("dataset", "my_dataset")])
    input_params = get_input_parameters(query_params=query_params)
    assert input_params
    assert input_params.input_type == "dataset"

    query_params = QueryParams([("dataset", "my_dataset"), ("config", "my_config")])
    input_params = get_input_parameters(query_params=query_params)
    assert input_params
    assert input_params.input_type == "config"

    query_params = QueryParams([("dataset", "my_dataset"), ("config", "my_config"), ("split", "my_split")])
    input_params = get_input_parameters(query_params=query_params)
    assert input_params
    assert input_params.input_type == "split"

    query_params = QueryParams([("dataset", "my_dataset"), ("split", "my_split")])
    with raises(MissingRequiredParameterError) as e:
        get_input_parameters(query_params=query_params)
    assert e.value.message == "Parameter 'config' is required"


def test_first_entry_from_steps() -> None:
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

    input_params = InputParameters(dataset=dataset, config=config, split=None)

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
    result = get_first_succeeded_cache_entry_from_steps(
        [step_whitout_error, step_with_error], input_params, init_processing_steps, app_config.common.hf_endpoint
    )
    assert result
    assert result["http_status"] == HTTPStatus.OK

    # succeeded result is returned even if first step failed
    result = get_first_succeeded_cache_entry_from_steps(
        [step_with_error, step_whitout_error], input_params, init_processing_steps, app_config.common.hf_endpoint
    )
    assert result
    assert result["http_status"] == HTTPStatus.OK

    # error result is returned if all steps failed
    result = get_first_succeeded_cache_entry_from_steps(
        [step_with_error, step_with_error], input_params, init_processing_steps, app_config.common.hf_endpoint
    )
    assert result
    assert result["http_status"] == HTTPStatus.INTERNAL_SERVER_ERROR

    # peding job returns None
    queue = Queue()
    queue.upsert_job(job_type="/splits", dataset=dataset, config=config, force=True)
    non_existent_step = graph.get_step("/splits")
    assert not get_first_succeeded_cache_entry_from_steps(
        [non_existent_step], input_params, init_processing_steps, app_config.common.hf_endpoint
    )


def test_get_response_from_cache_entry() -> None:
    # raises exception if no cache entry found
    with raises(ResponseNotReadyError) as e:
        get_response_from_cache_entry(result=None)
    assert e.value.message == "The server is busier than usual and the response is not ready yet. Please retry later."

    # returns OK response
    cache_entry = CacheEntry(
        http_status=HTTPStatus.OK,
        error_code=None,
        worker_version="worker_version",
        dataset_git_revision="git_version",
        content={},
    )

    response = get_response_from_cache_entry(cache_entry)
    assert response
    assert "X-Error-Code" not in response.headers

    # returns ERROR response
    error_code = "error_code"
    cache_entry = CacheEntry(
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
        error_code=error_code,
        worker_version="worker_version",
        dataset_git_revision="git_version",
        content={},
    )

    response = get_response_from_cache_entry(cache_entry)
    assert response
    assert response.headers["X-Error-Code"] == error_code
