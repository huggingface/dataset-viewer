# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import replace
from http import HTTPStatus

import pytest
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
from libcommon.simple_cache import DoesNotExist, get_response

from datasets_based.config import AppConfig
from datasets_based.workers.split_names import SplitNamesWorker

from ..fixtures.hub import HubDatasets, get_default_config_split


def get_worker(
    dataset: str,
    config: str,
    app_config: AppConfig,
    force: bool = False,
) -> SplitNamesWorker:
    return SplitNamesWorker(
        job_info={
            "type": SplitNamesWorker.get_job_type(),
            "dataset": dataset,
            "config": config,
            "split": None,
            "job_id": "job_id",
            "force": force,
            "priority": Priority.NORMAL,
        },
        app_config=app_config,
        processing_step=ProcessingStep(
            endpoint=SplitNamesWorker.get_job_type(),
            input_type="config",
            requires=None,
            required_by_dataset_viewer=False,
            parent=None,
            ancestors=[],
            children=[],
        ),
    )


def test_process(app_config: AppConfig, hub_public_csv: str) -> None:
    dataset, config, _ = get_default_config_split(hub_public_csv)
    worker = get_worker(dataset, config, app_config)
    assert worker.process() is True
    cached_response = get_response(kind=worker.processing_step.cache_kind, dataset=hub_public_csv, config=config)
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None
    assert cached_response["worker_version"] == worker.get_version()
    assert cached_response["dataset_git_revision"] is not None
    assert cached_response["error_code"] is None
    content = cached_response["content"]
    assert len(content["split_names"]) == 1


def test_doesnotexist(app_config: AppConfig) -> None:
    dataset = "doesnotexist"
    config = "some_config"
    worker = get_worker(dataset, config, app_config)
    assert worker.process() is False
    with pytest.raises(DoesNotExist):
        get_response(kind=worker.processing_step.cache_kind, dataset=dataset, config=config)


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
        ("does_not_exist", False, "SplitNamesError", "FileNotFoundError"),
        ("gated", False, "SplitNamesError", "FileNotFoundError"),
        ("private", False, "SplitNamesError", "FileNotFoundError"),
    ],
)
def test_compute_split_names_response(
    hub_datasets: HubDatasets, name: str, use_token: bool, error_code: str, cause: str, app_config: AppConfig
) -> None:
    dataset, config, _ = get_default_config_split(hub_datasets[name]["name"])
    worker = get_worker(dataset, config, app_config)
    expected_configs_response = hub_datasets[name]["split_names_response"]
    worker = get_worker(
        dataset,
        config,
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
