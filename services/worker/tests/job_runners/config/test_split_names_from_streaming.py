# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import replace
from http import HTTPStatus
from typing import Callable
from unittest.mock import Mock

import pytest
from libcommon.constants import PROCESSING_STEP_SPLIT_NAMES_FROM_DATASET_INFO_VERSION
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import DoesNotExist, get_response, upsert_response

from worker.config import AppConfig
from worker.job_runners.config.split_names_from_streaming import (
    SplitNamesFromStreamingJobRunner,
)
from worker.resources import LibrariesResource

from ...fixtures.hub import HubDatasets, get_default_config_split

GetJobRunner = Callable[[str, str, AppConfig, bool], SplitNamesFromStreamingJobRunner]


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
        force: bool = False,
    ) -> SplitNamesFromStreamingJobRunner:
        processing_step_name = SplitNamesFromStreamingJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            {
                "dataset-level": {"input_type": "dataset"},
                processing_step_name: {
                    "input_type": "dataset",
                    "job_runner_version": SplitNamesFromStreamingJobRunner.get_job_runner_version(),
                    "triggered_by": "dataset-level",
                },
            }
        )
        return SplitNamesFromStreamingJobRunner(
            job_info={
                "type": SplitNamesFromStreamingJobRunner.get_job_type(),
                "dataset": dataset,
                "config": config,
                "split": None,
                "job_id": "job_id",
                "force": force,
                "priority": Priority.NORMAL,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
            processing_graph=processing_graph,
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


def test_process(app_config: AppConfig, get_job_runner: GetJobRunner, hub_public_csv: str) -> None:
    dataset, config, _ = get_default_config_split(hub_public_csv)
    job_runner = get_job_runner(dataset, config, app_config, False)
    assert job_runner.process()
    cached_response = get_response(kind=job_runner.processing_step.cache_kind, dataset=hub_public_csv, config=config)
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None
    assert cached_response["job_runner_version"] == job_runner.get_job_runner_version()
    assert cached_response["dataset_git_revision"] is not None
    assert cached_response["error_code"] is None
    content = cached_response["content"]
    assert len(content["splits"]) == 1


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "doesnotexist"
    config = "some_config"
    job_runner = get_job_runner(dataset, config, app_config, False)
    assert not job_runner.process()
    with pytest.raises(DoesNotExist):
        get_response(kind=job_runner.processing_step.cache_kind, dataset=dataset, config=config)


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
    hub_datasets: HubDatasets,
    get_job_runner: GetJobRunner,
    name: str,
    use_token: bool,
    error_code: str,
    cause: str,
    app_config: AppConfig,
) -> None:
    dataset, config, _ = get_default_config_split(hub_datasets[name]["name"])
    expected_configs_response = hub_datasets[name]["splits_response"]
    job_runner = get_job_runner(
        dataset,
        config,
        app_config if use_token else replace(app_config, common=replace(app_config.common, hf_token=None)),
        False,
    )
    job_runner.get_dataset_git_revision = Mock(return_value="1.0.0")  # type: ignore
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


@pytest.mark.parametrize(
    "dataset_info_response_status,dataset_git_revision,error_code",
    [
        (HTTPStatus.OK, "CURRENT_GIT_REVISION", "ResponseAlreadyComputedError"),
        (HTTPStatus.INTERNAL_SERVER_ERROR, "CURRENT_GIT_REVISION", "SplitNamesFromStreamingError"),
        (HTTPStatus.OK, "DIFFERENT_GIT_REVISION", "SplitNamesFromStreamingError"),
    ],
)
def test_response_already_computed(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset_info_response_status: HTTPStatus,
    dataset_git_revision: str,
    error_code: str,
) -> None:
    dataset = "dataset"
    config = "config"
    current_dataset_git_revision = "CURRENT_GIT_REVISION"
    upsert_response(
        kind="/split-names-from-dataset-info",
        dataset=dataset,
        config=config,
        content={},
        dataset_git_revision=dataset_git_revision,
        job_runner_version=PROCESSING_STEP_SPLIT_NAMES_FROM_DATASET_INFO_VERSION,
        progress=1.0,
        http_status=dataset_info_response_status,
    )
    job_runner = get_job_runner(dataset, config, app_config, False)
    job_runner.get_dataset_git_revision = Mock(return_value=current_dataset_git_revision)  # type: ignore
    with pytest.raises(CustomError) as exc_info:
        job_runner.compute()
    assert exc_info.value.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert exc_info.value.code == error_code
