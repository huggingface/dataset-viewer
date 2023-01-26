# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import replace
from http import HTTPStatus

import pytest
from datasets.packaged_modules import csv
from libcommon.exceptions import CustomError
from libcommon.queue import Priority
from libcommon.simple_cache import DoesNotExist, get_response

from datasets_based.config import AppConfig, FirstRowsConfig
from datasets_based.workers.first_rows import FirstRowsWorker, get_json_size

from ..fixtures.hub import HubDatasets, get_default_config_split


def get_worker(
    dataset: str,
    config: str,
    split: str,
    app_config: AppConfig,
    first_rows_config: FirstRowsConfig,
    force: bool = False,
) -> FirstRowsWorker:
    return FirstRowsWorker(
        job_info={
            "type": FirstRowsWorker.get_job_type(),
            "dataset": dataset,
            "config": config,
            "split": split,
            "job_id": "job_id",
            "force": force,
            "priority": Priority.NORMAL,
        },
        app_config=app_config,
        first_rows_config=first_rows_config,
    )


def test_should_skip_job(app_config: AppConfig, first_rows_config: FirstRowsConfig, hub_public_csv: str) -> None:
    dataset, config, split = get_default_config_split(hub_public_csv)
    worker = get_worker(dataset, config, split, app_config, first_rows_config)
    assert worker.should_skip_job() is False
    # we add an entry to the cache
    worker.process()
    assert worker.should_skip_job() is True
    worker = get_worker(dataset, config, split, app_config, first_rows_config, force=True)
    assert worker.should_skip_job() is False


def test_compute(app_config: AppConfig, first_rows_config: FirstRowsConfig, hub_public_csv: str) -> None:
    dataset, config, split = get_default_config_split(hub_public_csv)
    worker = get_worker(dataset, config, split, app_config, first_rows_config)
    assert worker.process() is True
    cached_response = get_response(kind=worker.processing_step.cache_kind, dataset=dataset, config=config, split=split)
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None
    assert cached_response["worker_version"] == worker.get_version()
    assert cached_response["dataset_git_revision"] is not None
    content = cached_response["content"]
    assert content["features"][0]["feature_idx"] == 0
    assert content["features"][0]["name"] == "col_1"
    assert content["features"][0]["type"]["_type"] == "Value"
    assert content["features"][0]["type"]["dtype"] == "int64"  # <---|
    assert content["features"][1]["type"]["dtype"] == "int64"  # <---|- auto-detected by the datasets library
    assert content["features"][2]["type"]["dtype"] == "float64"  # <-|


def test_doesnotexist(app_config: AppConfig, first_rows_config: FirstRowsConfig) -> None:
    dataset = "doesnotexist"
    dataset, config, split = get_default_config_split(dataset)
    worker = get_worker(dataset, config, split, app_config, first_rows_config)
    assert worker.process() is False
    with pytest.raises(DoesNotExist):
        get_response(kind=worker.processing_step.cache_kind, dataset=dataset, config=config, split=split)


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
    first_rows_config: FirstRowsConfig,
) -> None:
    # temporary patch to remove the effect of
    # https://github.com/huggingface/datasets/issues/4875#issuecomment-1280744233
    # note: it fixes the tests, but it does not fix the bug in the "real world"
    if hasattr(csv, "_patched_for_streaming") and csv._patched_for_streaming:  # type: ignore
        csv._patched_for_streaming = False  # type: ignore

    dataset = hub_datasets[name]["name"]
    expected_first_rows_response = hub_datasets[name]["first_rows_response"]
    dataset, config, split = get_default_config_split(dataset)
    worker = get_worker(
        dataset,
        config,
        split,
        app_config if use_token else replace(app_config, common=replace(app_config.common, hf_token=None)),
        first_rows_config,
    )
    if error_code is None:
        result = worker.compute()
        assert result == expected_first_rows_response
        return
    with pytest.raises(CustomError) as exc_info:
        worker.compute()
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
    first_rows_config: FirstRowsConfig,
    name: str,
    rows_max_bytes: int,
    successful_truncation: bool,
) -> None:
    dataset, config, split = get_default_config_split(hub_datasets[name]["name"])
    worker = get_worker(
        dataset,
        config,
        split,
        app_config=replace(app_config, common=replace(app_config.common, hf_token=None)),
        first_rows_config=replace(
            first_rows_config,
            max_number=1_000_000,
            min_number=10,
            max_bytes=rows_max_bytes,
            min_cell_bytes=10,
        ),
    )
    response = worker.compute()
    print(get_json_size(response))
    assert (get_json_size(response) <= rows_max_bytes) is successful_truncation
