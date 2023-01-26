# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import replace
from http import HTTPStatus

import pytest
from libcommon.exceptions import CustomError
from libcommon.simple_cache import DoesNotExist, get_response

from datasets_based.config import AppConfig
from datasets_based.workers.config_names import ConfigNamesWorker

from ..fixtures.hub import HubDatasets


def get_worker(
    dataset: str,
    app_config: AppConfig,
    force: bool = False,
) -> ConfigNamesWorker:
    return ConfigNamesWorker(
        job_info={
            "type": ConfigNamesWorker.get_job_type(),
            "dataset": dataset,
            "config": None,
            "split": None,
            "job_id": "job_id",
            "force": force,
        },
        app_config=app_config,
    )


def test_should_skip_job(app_config: AppConfig, hub_public_csv: str) -> None:
    dataset = hub_public_csv
    worker = get_worker(dataset, app_config)
    assert worker.should_skip_job() is False
    # we add an entry to the cache
    worker.process()
    assert worker.should_skip_job() is True
    worker = get_worker(dataset, app_config, force=True)
    assert worker.should_skip_job() is False


def test_process(app_config: AppConfig, hub_public_csv: str) -> None:
    dataset = hub_public_csv
    worker = get_worker(dataset, app_config)
    assert worker.process() is True
    cached_response = get_response(kind=worker.processing_step.cache_kind, dataset=hub_public_csv)
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None
    assert cached_response["worker_version"] == worker.get_version()
    assert cached_response["dataset_git_revision"] is not None
    assert cached_response["error_code"] is None
    content = cached_response["content"]
    assert len(content["config_names"]) == 1


def test_doesnotexist(app_config: AppConfig) -> None:
    dataset = "doesnotexist"
    worker = get_worker(dataset, app_config)
    assert worker.process() is False
    with pytest.raises(DoesNotExist):
        get_response(kind=worker.processing_step.cache_kind, dataset=dataset)


@pytest.mark.parametrize(
    "name,use_token,error_code,cause",
    [
        ("public", False, None, None),
        ("audio", False, None, None),
        ("gated", True, None, None),
        ("private", True, None, None),
        ("empty", False, "EmptyDatasetError", "EmptyDatasetError"),
        # should we really test the following cases?
        # The assumption is that the dataset exists and is accessible with the token
        ("does_not_exist", False, "ConfigNamesError", "FileNotFoundError"),
        ("gated", False, "ConfigNamesError", "FileNotFoundError"),
        ("private", False, "ConfigNamesError", "FileNotFoundError"),
    ],
)
def test_compute_splits_response_simple_csv(
    hub_datasets: HubDatasets, name: str, use_token: bool, error_code: str, cause: str, app_config: AppConfig
) -> None:
    dataset = hub_datasets[name]["name"]
    expected_configs_response = hub_datasets[name]["config_names_response"]
    worker = get_worker(
        dataset,
        app_config if use_token else replace(app_config, common=replace(app_config.common, hf_token=None)),
    )
    if error_code is None:
        result = worker.compute()
        assert result == expected_configs_response
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
