from http import HTTPStatus
from typing import List

import pytest
from libcache.simple_cache import _clean_cache_database, upsert_response
from libcommon.processing_steps import ProcessingStep, Parameters

from api.config import AppConfig
from api.routes.valid import get_valid, is_valid

dataset_step = ProcessingStep(endpoint="/dataset-step", parameters=Parameters.DATASET, dependencies=[])
split_step = ProcessingStep(endpoint="/split-step", parameters=Parameters.SPLIT, dependencies=[])


@pytest.fixture(autouse=True)
def clean_mongo_databases(app_config: AppConfig) -> None:
    _clean_cache_database()


@pytest.mark.parametrize(
    "processing_steps_for_valid,expected_is_valid",
    [
        ([], True),
        ([dataset_step], False),
        ([dataset_step, split_step], False),
    ],
)
def test_empty(processing_steps_for_valid: List[ProcessingStep], expected_is_valid: bool) -> None:
    assert get_valid(processing_steps_for_valid=processing_steps_for_valid) == []
    assert is_valid(dataset="dataset", processing_steps_for_valid=processing_steps_for_valid) is expected_is_valid


@pytest.mark.parametrize(
    "processing_steps_for_valid,expected_is_valid,expected_valid",
    [
        ([], True, []),
        ([dataset_step], True, ["dataset"]),
        ([split_step], False, []),
        ([dataset_step, split_step], False, []),
    ],
)
def test_one_step(
    processing_steps_for_valid: List[ProcessingStep], expected_is_valid: bool, expected_valid: List[str]
) -> None:
    dataset = "dataset"
    upsert_response(kind=dataset_step.cache_kind, dataset=dataset, content={}, http_status=HTTPStatus.OK)
    assert get_valid(processing_steps_for_valid=processing_steps_for_valid) == expected_valid
    assert is_valid(dataset=dataset, processing_steps_for_valid=processing_steps_for_valid) is expected_is_valid


@pytest.mark.parametrize(
    "processing_steps_for_valid,expected_is_valid,expected_valid",
    [
        ([], True, []),
        ([dataset_step], True, ["dataset"]),
        ([split_step], True, ["dataset"]),
        ([dataset_step, split_step], True, ["dataset"]),
    ],
)
def test_two_steps(
    processing_steps_for_valid: List[ProcessingStep], expected_is_valid: bool, expected_valid: List[str]
) -> None:
    dataset = "dataset"
    config = "config"
    split = "split"
    upsert_response(kind=dataset_step.cache_kind, dataset=dataset, content={}, http_status=HTTPStatus.OK)
    upsert_response(
        kind=split_step.cache_kind, dataset=dataset, config=config, split=split, content={}, http_status=HTTPStatus.OK
    )
    assert get_valid(processing_steps_for_valid=processing_steps_for_valid) == expected_valid
    assert is_valid(dataset=dataset, processing_steps_for_valid=processing_steps_for_valid) is expected_is_valid


def test_errors() -> None:
    processing_steps_for_valid = [dataset_step]
    dataset_a = "dataset_a"
    dataset_b = "dataset_b"
    dataset_c = "dataset_c"
    upsert_response(kind=dataset_step.cache_kind, dataset=dataset_a, content={}, http_status=HTTPStatus.OK)
    upsert_response(kind=dataset_step.cache_kind, dataset=dataset_b, content={}, http_status=HTTPStatus.OK)
    upsert_response(
        kind=dataset_step.cache_kind, dataset=dataset_c, content={}, http_status=HTTPStatus.INTERNAL_SERVER_ERROR
    )
    assert get_valid(processing_steps_for_valid=processing_steps_for_valid) == [dataset_a, dataset_b]
    assert is_valid(dataset=dataset_a, processing_steps_for_valid=processing_steps_for_valid) is True
    assert is_valid(dataset=dataset_b, processing_steps_for_valid=processing_steps_for_valid) is True
    assert is_valid(dataset=dataset_c, processing_steps_for_valid=processing_steps_for_valid) is False
