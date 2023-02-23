# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus

from libcommon.config import ProcessingGraphConfig
from libcommon.processing_graph import ProcessingGraph
from libcommon.simple_cache import upsert_response
from pytest import raises
from starlette.datastructures import QueryParams

from api.config import EndpointConfig
from api.routes.endpoint import (
    EndpointsDefinition,
    InputParams,
    first_entry_from_steps,
    get_params,
)
from api.utils import MissingRequiredParameterError


def test_endpoints_definition() -> None:
    config = ProcessingGraphConfig()
    graph = ProcessingGraph(config.specification)
    endpoint_config = EndpointConfig.from_env()

    endpoints_definition = EndpointsDefinition(graph, endpoint_config)
    assert endpoints_definition

    definition = endpoints_definition.definition
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


def test_get_params() -> None:
    query_params = QueryParams([("config", None)])
    with raises(MissingRequiredParameterError) as e:
        get_params(query_params=query_params)
    assert e.value.message == "Parameter 'dataset' is required"

    query_params = QueryParams([("dataset", "my_dataset")])
    input_params = get_params(query_params=query_params)
    assert input_params
    assert input_params.input_type == "dataset"

    query_params = QueryParams([("dataset", "my_dataset"), ("config", "my_config")])
    input_params = get_params(query_params=query_params)
    assert input_params
    assert input_params.input_type == "config"

    query_params = QueryParams([("dataset", "my_dataset"), ("config", "my_config"), ("split", "my_split")])
    input_params = get_params(query_params=query_params)
    assert input_params
    assert input_params.input_type == "split"

    query_params = QueryParams([("dataset", "my_dataset"), ("split", "my_split")])
    with raises(MissingRequiredParameterError) as e:
        get_params(query_params=query_params)
    assert e.value.message == "Parameter 'config' is required"


def test_first_entry_from_steps() -> None:
    dataset = "dataset"
    config = "config"

    graph_config = ProcessingGraphConfig()
    graph = ProcessingGraph(graph_config.specification)

    cache_with_error = "/split-names-from-streaming"
    cache_without_error = "/split-names-from-dataset-info"

    step_with_error = graph.get_step(cache_with_error)
    step_whitout_error = graph.get_step(cache_without_error)

    input_params = InputParams(input_type="config", dataset=dataset, config=config, split=None)

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

    result = first_entry_from_steps([step_with_error, step_whitout_error], input_params)
    assert result
    assert result["http_status"] == HTTPStatus.INTERNAL_SERVER_ERROR

    result = first_entry_from_steps([step_whitout_error, step_with_error], input_params)
    assert result
    assert result["http_status"] == HTTPStatus.OK

    non_existent_step = graph.get_step("/splits")
    assert not first_entry_from_steps([non_existent_step], input_params)
