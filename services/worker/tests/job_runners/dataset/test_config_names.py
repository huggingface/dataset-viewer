# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import replace
from typing import Callable
from unittest.mock import patch

import pytest
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.job_runners.dataset.config_names import DatasetConfigNamesJobRunner
from worker.resources import LibrariesResource

from ...fixtures.hub import HubDatasetTest

GetJobRunner = Callable[[str, AppConfig], DatasetConfigNamesJobRunner]


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetConfigNamesJobRunner:
        processing_step_name = DatasetConfigNamesJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            {
                processing_step_name: {
                    "input_type": "dataset",
                    "job_runner_version": DatasetConfigNamesJobRunner.get_job_runner_version(),
                }
            }
        )
        return DatasetConfigNamesJobRunner(
            job_info={
                "type": DatasetConfigNamesJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": None,
                    "split": None,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 50,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


def test_compute(app_config: AppConfig, hub_public_csv: str, get_job_runner: GetJobRunner) -> None:
    dataset = hub_public_csv
    job_runner = get_job_runner(dataset, app_config)
    response = job_runner.compute()
    content = response.content
    assert len(content["config_names"]) == 1


@pytest.mark.parametrize(
    "max_number_of_configs,error_code",
    [
        (1, "DatasetWithTooManyConfigsError"),
        (2, None),
        (3, None),
    ],
)
def test_compute_too_many_configs(
    app_config: AppConfig, get_job_runner: GetJobRunner, max_number_of_configs: int, error_code: str
) -> None:
    dataset = "dataset"
    configs = ["config_1", "config_2"]
    job_runner = get_job_runner(
        dataset,
        replace(app_config, config_names=replace(app_config.config_names, max_number=max_number_of_configs)),
    )

    with patch("worker.job_runners.dataset.config_names.get_dataset_config_names", return_value=configs):
        if error_code:
            with pytest.raises(CustomError) as exc_info:
                job_runner.compute()
            assert exc_info.value.code == error_code
        else:
            assert job_runner.compute() is not None


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
    hub_responses_public: HubDatasetTest,
    hub_reponses_audio: HubDatasetTest,
    hub_reponses_gated: HubDatasetTest,
    hub_reponses_private: HubDatasetTest,
    hub_reponses_empty: HubDatasetTest,
    hub_reponses_does_not_exist: HubDatasetTest,
    get_job_runner: GetJobRunner,
    name: str,
    use_token: bool,
    error_code: str,
    cause: str,
    app_config: AppConfig,
) -> None:
    hub_datasets = {
        "public": hub_responses_public,
        "audio": hub_reponses_audio,
        "gated": hub_reponses_gated,
        "private": hub_reponses_private,
        "empty": hub_reponses_empty,
        "does_not_exist": hub_reponses_does_not_exist,
    }
    dataset = hub_datasets[name]["name"]
    expected_configs_response = hub_datasets[name]["config_names_response"]
    job_runner = get_job_runner(
        dataset,
        app_config if use_token else replace(app_config, common=replace(app_config.common, hf_token=None)),
    )
    if error_code is None:
        result = job_runner.compute().content
        assert result == expected_configs_response
        return

    with pytest.raises(CustomError) as exc_info:
        job_runner.compute()
    assert exc_info.value.code == error_code
    assert exc_info.value.cause_exception == cause
    if exc_info.value.disclose_cause:
        response = exc_info.value.as_response()
        assert set(response.keys()) == {"error", "cause_exception", "cause_message", "cause_traceback"}
        response_dict = dict(response)
        # ^ to remove mypy warnings
        assert response_dict["cause_exception"] == cause
        assert isinstance(response_dict["cause_traceback"], list)
        assert response_dict["cause_traceback"][0] == "Traceback (most recent call last):\n"
