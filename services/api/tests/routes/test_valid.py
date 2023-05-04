from http import HTTPStatus
from typing import List

import pytest
from libcommon.processing_graph import ProcessingGraph, ProcessingGraphSpecification
from libcommon.simple_cache import _clean_cache_database, upsert_response

from api.config import AppConfig
from api.routes.valid import get_valid, is_valid

dataset_step = "dataset-step"
config_step = "config-step"
split_step = "split-step"

step_1 = "step-1"
step_2 = "step-2"


@pytest.fixture(autouse=True)
def clean_mongo_databases(app_config: AppConfig) -> None:
    _clean_cache_database()


@pytest.mark.parametrize(
    "processing_graph_specification,expected_is_valid",
    [
        ({}, True),
        ({step_1: {}}, True),
        ({step_1: {"required_by_dataset_viewer": True}}, False),
    ],
)
def test_empty(processing_graph_specification: ProcessingGraphSpecification, expected_is_valid: bool) -> None:
    processing_graph = ProcessingGraph(processing_graph_specification)
    assert get_valid(processing_graph=processing_graph) == []
    assert is_valid(dataset="dataset", processing_graph=processing_graph) is expected_is_valid


@pytest.mark.parametrize(
    "processing_graph_specification,expected_is_valid,expected_valid",
    [
        ({step_1: {}}, True, []),
        ({step_1: {"required_by_dataset_viewer": True}}, True, ["dataset"]),
        ({step_1: {}, step_2: {"required_by_dataset_viewer": True}}, False, []),
        ({step_1: {"required_by_dataset_viewer": True}, step_2: {"required_by_dataset_viewer": True}}, False, []),
    ],
)
def test_one_step(
    processing_graph_specification: ProcessingGraphSpecification, expected_is_valid: bool, expected_valid: List[str]
) -> None:
    dataset = "dataset"
    processing_graph = ProcessingGraph(processing_graph_specification)
    processing_step = processing_graph.get_processing_step(step_1)
    upsert_response(kind=processing_step.cache_kind, dataset=dataset, content={}, http_status=HTTPStatus.OK)
    assert get_valid(processing_graph=processing_graph) == expected_valid
    assert is_valid(dataset=dataset, processing_graph=processing_graph) is expected_is_valid


@pytest.mark.parametrize(
    "processing_graph_specification,expected_is_valid,expected_valid",
    [
        ({dataset_step: {}, config_step: {"input_type": "config"}, split_step: {"input_type": "split"}}, True, []),
        (
            {
                dataset_step: {"required_by_dataset_viewer": True},
                config_step: {"input_type": "config"},
                split_step: {"input_type": "split"},
            },
            True,
            ["dataset"],
        ),
        (
            {
                dataset_step: {},
                config_step: {"input_type": "config", "required_by_dataset_viewer": True},
                split_step: {"input_type": "split"},
            },
            True,
            ["dataset"],
        ),
        (
            {
                dataset_step: {},
                config_step: {"input_type": "config"},
                split_step: {"input_type": "split", "required_by_dataset_viewer": True},
            },
            True,
            ["dataset"],
        ),
        (
            {
                dataset_step: {"required_by_dataset_viewer": True},
                config_step: {"input_type": "config", "required_by_dataset_viewer": True},
                split_step: {"input_type": "split", "required_by_dataset_viewer": True},
            },
            True,
            ["dataset"],
        ),
    ],
)
def test_three_steps(
    processing_graph_specification: ProcessingGraphSpecification, expected_is_valid: bool, expected_valid: List[str]
) -> None:
    dataset = "dataset"
    config = "config"
    split = "split"
    processing_graph = ProcessingGraph(processing_graph_specification)
    upsert_response(
        kind=processing_graph.get_processing_step(dataset_step).cache_kind,
        dataset=dataset,
        content={},
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind=processing_graph.get_processing_step(config_step).cache_kind,
        dataset=dataset,
        config=config,
        content={},
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind=processing_graph.get_processing_step(split_step).cache_kind,
        dataset=dataset,
        config=config,
        split=split,
        content={},
        http_status=HTTPStatus.OK,
    )
    assert get_valid(processing_graph=processing_graph) == expected_valid
    assert is_valid(dataset=dataset, processing_graph=processing_graph) is expected_is_valid


def test_errors() -> None:
    processing_graph = ProcessingGraph({dataset_step: {"required_by_dataset_viewer": True}})
    dataset_a = "dataset_a"
    dataset_b = "dataset_b"
    dataset_c = "dataset_c"
    cache_kind = processing_graph.get_processing_step(dataset_step).cache_kind
    upsert_response(kind=cache_kind, dataset=dataset_a, content={}, http_status=HTTPStatus.OK)
    upsert_response(kind=cache_kind, dataset=dataset_b, content={}, http_status=HTTPStatus.OK)
    upsert_response(kind=cache_kind, dataset=dataset_c, content={}, http_status=HTTPStatus.INTERNAL_SERVER_ERROR)
    assert get_valid(processing_graph=processing_graph) == [dataset_a, dataset_b]
    assert is_valid(dataset=dataset_a, processing_graph=processing_graph)
    assert is_valid(dataset=dataset_b, processing_graph=processing_graph)
    assert not is_valid(dataset=dataset_c, processing_graph=processing_graph)
