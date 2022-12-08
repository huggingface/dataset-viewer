# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus

import pytest
from datasets.packaged_modules import csv
from libcommon.exceptions import CustomError
from libcommon.simple_cache import DoesNotExist, get_response

from datasets_based.config import AppConfig
from datasets_based.workers.first_rows import (
    FirstRowsWorker,
    compute_first_rows_response,
    get_json_size,
)

from ..fixtures.hub import HubDatasets, get_default_config_split


@pytest.fixture
def worker(app_config: AppConfig) -> FirstRowsWorker:
    return FirstRowsWorker(app_config=app_config)


def should_skip_job(worker: FirstRowsWorker, hub_public_csv: str) -> None:
    dataset, config, split = get_default_config_split(hub_public_csv)
    assert worker.should_skip_job(dataset=dataset, config=config, split=split) is False
    # we add an entry to the cache
    worker.process(dataset=dataset, config=config, split=split)
    assert worker.should_skip_job(dataset=dataset, config=config, split=split) is True
    assert worker.should_skip_job(dataset=dataset, config=config, split=split, force=False) is False


def test_compute(worker: FirstRowsWorker, hub_public_csv: str) -> None:
    dataset, config, split = get_default_config_split(hub_public_csv)
    assert worker.process(dataset=dataset, config=config, split=split) is True
    cached_response = get_response(kind=worker.processing_step.cache_kind, dataset=dataset, config=config, split=split)
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None
    assert cached_response["worker_version"] == worker.version
    assert cached_response["dataset_git_revision"] is not None
    content = cached_response["content"]
    assert content["features"][0]["feature_idx"] == 0
    assert content["features"][0]["name"] == "col_1"
    assert content["features"][0]["type"]["_type"] == "Value"
    assert content["features"][0]["type"]["dtype"] == "int64"  # <---|
    assert content["features"][1]["type"]["dtype"] == "int64"  # <---|- auto-detected by the datasets library
    assert content["features"][2]["type"]["dtype"] == "float64"  # <-|


def test_doesnotexist(worker: FirstRowsWorker) -> None:
    dataset = "doesnotexist"
    dataset, config, split = get_default_config_split(dataset)
    assert worker.process(dataset=dataset, config=config, split=split) is False
    with pytest.raises(DoesNotExist):
        get_response(kind=worker.processing_step.cache_kind, dataset=dataset, config=config, split=split)


def test_process_job(worker: FirstRowsWorker, hub_public_csv: str) -> None:
    dataset, config, split = get_default_config_split(hub_public_csv)
    worker.queue.add_job(dataset=dataset, config=config, split=split)
    result = worker.process_next_job()
    assert result is True


@pytest.mark.parametrize(
    "name,use_token,error_code,cause",
    [
        ("public", False, None, None),
        ("audio", False, None, None),
        ("image", False, None, None),
        ("images_list", False, None, None),
        ("jsonl", False, None, None),
        ("gated", True, None, None),
        ("private", True, None, None),
        ("empty", False, "EmptyDatasetError", "EmptyDatasetError"),
        # should we really test the following cases?
        # The assumption is that the dataset exists and is accessible with the token
        ("does_not_exist", False, "SplitsNamesError", "FileNotFoundError"),
        ("gated", False, "SplitsNamesError", "FileNotFoundError"),
        ("private", False, "SplitsNamesError", "FileNotFoundError"),
    ],
)
def test_number_rows(
    hub_datasets: HubDatasets,
    name: str,
    use_token: bool,
    error_code: str,
    cause: str,
    app_config: AppConfig,
) -> None:
    # temporary patch to remove the effect of
    # https://github.com/huggingface/datasets/issues/4875#issuecomment-1280744233
    # note: it fixes the tests, but it does not fix the bug in the "real world"
    if hasattr(csv, "_patched_for_streaming") and csv._patched_for_streaming:  # type: ignore
        csv._patched_for_streaming = False  # type: ignore

    dataset = hub_datasets[name]["name"]
    expected_first_rows_response = hub_datasets[name]["first_rows_response"]
    dataset, config, split = get_default_config_split(dataset)
    if error_code is None:
        result = compute_first_rows_response(
            dataset=dataset,
            config=config,
            split=split,
            assets_base_url=app_config.first_rows.assets.base_url,
            assets_directory=app_config.first_rows.assets.storage_directory,
            hf_token=app_config.common.hf_token if use_token else None,
            max_size_fallback=app_config.first_rows.fallback_max_dataset_size,
            rows_max_number=app_config.first_rows.max_number,
            rows_min_number=app_config.first_rows.min_number,
            rows_max_bytes=app_config.first_rows.max_bytes,
            min_cell_bytes=app_config.first_rows.min_cell_bytes,
        )
        assert result == expected_first_rows_response
        return
    with pytest.raises(CustomError) as exc_info:
        compute_first_rows_response(
            dataset=dataset,
            config=config,
            split=split,
            assets_base_url=app_config.first_rows.assets.base_url,
            assets_directory=app_config.first_rows.assets.storage_directory,
            hf_token=app_config.common.hf_token if use_token else None,
            max_size_fallback=app_config.first_rows.fallback_max_dataset_size,
            rows_max_number=app_config.first_rows.max_number,
            rows_min_number=app_config.first_rows.min_number,
            rows_max_bytes=app_config.first_rows.max_bytes,
            min_cell_bytes=app_config.first_rows.min_cell_bytes,
        )
    assert exc_info.value.code == error_code
    if cause is None:
        assert exc_info.value.disclose_cause is False
        assert exc_info.value.cause_exception is None
    else:
        assert exc_info.value.disclose_cause is True
        assert exc_info.value.cause_exception == cause
        response = exc_info.value.as_response()
        assert set(response.keys()) == {"error", "cause_exception", "cause_message", "cause_traceback"}
        response_dict = dict(response)
        # ^ to remove mypy warnings
        assert response_dict["cause_exception"] == cause
        assert isinstance(response_dict["cause_traceback"], list)
        assert response_dict["cause_traceback"][0] == "Traceback (most recent call last):\n"


@pytest.mark.parametrize(
    "name,rows_max_bytes,successful_truncation",
    [
        # not-truncated public response is 687 bytes
        ("public", 10, False),  # too small limit, even with truncation
        ("public", 1_000, True),  # not truncated
        # not-truncated big response is 5_885_989 bytes
        ("big", 10, False),  # too small limit, even with truncation
        ("big", 1_000, True),  # truncated successfully
        ("big", 10_000_000, True),  # not truncated
    ],
)
def test_truncation(
    hub_datasets: HubDatasets,
    app_config: AppConfig,
    name: str,
    rows_max_bytes: int,
    successful_truncation: bool,
) -> None:
    dataset, config, split = get_default_config_split(hub_datasets[name]["name"])
    response = compute_first_rows_response(
        dataset=dataset,
        config=config,
        split=split,
        assets_base_url=app_config.first_rows.assets.base_url,
        assets_directory=app_config.first_rows.assets.storage_directory,
        hf_token=None,
        max_size_fallback=app_config.first_rows.fallback_max_dataset_size,
        rows_max_number=1_000_000,
        rows_min_number=10,
        rows_max_bytes=rows_max_bytes,
        min_cell_bytes=10,
    )
    print(get_json_size(response))
    assert (get_json_size(response) <= rows_max_bytes) is successful_truncation
