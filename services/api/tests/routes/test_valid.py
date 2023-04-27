from http import HTTPStatus
from typing import List

import pytest
from libcommon.processing_graph import ProcessingStep
from libcommon.simple_cache import _clean_cache_database, upsert_response

from api.config import AppConfig
from api.routes.valid import get_valid, is_valid

dataset_step = ProcessingStep(name="/dataset-step", input_type="dataset", job_runner_version=1)
config_step = ProcessingStep(name="/config-step", input_type="config", job_runner_version=1)
split_step = ProcessingStep(name="/split-step", input_type="split", job_runner_version=1)


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
        ([config_step], True, ["dataset"]),
        ([split_step], True, ["dataset"]),
        ([dataset_step, config_step, split_step], True, ["dataset"]),
    ],
)
def test_three_steps(
    processing_steps_for_valid: List[ProcessingStep], expected_is_valid: bool, expected_valid: List[str]
) -> None:
    dataset = "dataset"
    config = "config"
    split = "split"
    upsert_response(kind=dataset_step.cache_kind, dataset=dataset, content={}, http_status=HTTPStatus.OK)
    upsert_response(kind=config_step.cache_kind, dataset=dataset, config=config, content={}, http_status=HTTPStatus.OK)
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
    assert is_valid(dataset=dataset_a, processing_steps_for_valid=processing_steps_for_valid)
    assert is_valid(dataset=dataset_b, processing_steps_for_valid=processing_steps_for_valid)
    assert not is_valid(dataset=dataset_c, processing_steps_for_valid=processing_steps_for_valid)
