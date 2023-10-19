# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Callable
from dataclasses import replace
from http import HTTPStatus

import pytest
from libcommon.config import ProcessingGraphConfig
from libcommon.exceptions import CustomError, DatasetManualDownloadError
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.job_runners.config.split_names_from_streaming import (
    ConfigSplitNamesFromStreamingJobRunner,
    compute_split_names_from_streaming_response,
)
from worker.resources import LibrariesResource

from ...fixtures.hub import HubDatasetTest, get_default_config_split
from ..utils import REVISION_NAME

GetJobRunner = Callable[[str, str, AppConfig], ConfigSplitNamesFromStreamingJobRunner]


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigSplitNamesFromStreamingJobRunner:
        processing_step_name = ConfigSplitNamesFromStreamingJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            ProcessingGraphConfig(
                {
                    "dataset-level": {"input_type": "dataset"},
                    processing_step_name: {
                        "input_type": "dataset",
                        "job_runner_version": ConfigSplitNamesFromStreamingJobRunner.get_job_runner_version(),
                        "triggered_by": "dataset-level",
                    },
                }
            )
        )

        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        return ConfigSplitNamesFromStreamingJobRunner(
            job_info={
                "type": ConfigSplitNamesFromStreamingJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": REVISION_NAME,
                    "config": config,
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


def test_compute(app_config: AppConfig, get_job_runner: GetJobRunner, hub_public_csv: str) -> None:
    dataset = hub_public_csv
    config, _ = get_default_config_split()
    job_runner = get_job_runner(dataset, config, app_config)
    response = job_runner.compute()
    content = response.content
    assert len(content["splits"]) == 1


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
        ("does_not_exist", False, "SplitNamesFromStreamingError", "FileNotFoundError"),
        ("gated", False, "SplitNamesFromStreamingError", "FileNotFoundError"),
        ("private", False, "SplitNamesFromStreamingError", "FileNotFoundError"),
    ],
)
def test_compute_split_names_from_streaming_response(
    hub_responses_public: HubDatasetTest,
    hub_responses_audio: HubDatasetTest,
    hub_responses_gated: HubDatasetTest,
    hub_responses_private: HubDatasetTest,
    hub_responses_empty: HubDatasetTest,
    hub_responses_does_not_exist: HubDatasetTest,
    get_job_runner: GetJobRunner,
    name: str,
    use_token: bool,
    error_code: str,
    cause: str,
    app_config: AppConfig,
) -> None:
    hub_datasets = {
        "public": hub_responses_public,
        "audio": hub_responses_audio,
        "gated": hub_responses_gated,
        "private": hub_responses_private,
        "empty": hub_responses_empty,
        "does_not_exist": hub_responses_does_not_exist,
    }
    dataset = hub_datasets[name]["name"]
    config, _ = get_default_config_split()
    expected_configs_response = hub_datasets[name]["splits_response"]
    job_runner = get_job_runner(
        dataset,
        config,
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


def test_compute_split_names_from_streaming_response_raises(
    hub_public_manual_download: str, app_config: AppConfig
) -> None:
    with pytest.raises(DatasetManualDownloadError):
        compute_split_names_from_streaming_response(
            hub_public_manual_download, "default", hf_token=app_config.common.hf_token
        )
