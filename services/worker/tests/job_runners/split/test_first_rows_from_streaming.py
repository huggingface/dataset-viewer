# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import replace
from http import HTTPStatus
from typing import Callable
from unittest.mock import Mock

import pytest
from datasets.packaged_modules import csv
from libcommon.constants import PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_PARQUET_VERSION
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import DoesNotExist, get_response, upsert_response
from libcommon.storage import StrPath

from worker.config import AppConfig, FirstRowsConfig
from worker.job_runners.split.first_rows_from_streaming import (
    SplitFirstRowsFromStreamingJobRunner,
)
from worker.resources import LibrariesResource
from worker.utils import get_json_size

from ...fixtures.hub import HubDatasets, get_default_config_split

GetJobRunner = Callable[[str, str, str, AppConfig, FirstRowsConfig, bool], SplitFirstRowsFromStreamingJobRunner]


@pytest.fixture
def get_job_runner(
    assets_directory: StrPath,
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        split: str,
        app_config: AppConfig,
        first_rows_config: FirstRowsConfig,
        force: bool = False,
    ) -> SplitFirstRowsFromStreamingJobRunner:
        return SplitFirstRowsFromStreamingJobRunner(
            job_info={
                "type": SplitFirstRowsFromStreamingJobRunner.get_job_type(),
                "dataset": dataset,
                "config": config,
                "split": split,
                "job_id": "job_id",
                "force": force,
                "priority": Priority.NORMAL,
            },
            app_config=app_config,
            processing_step=ProcessingStep(
                name=SplitFirstRowsFromStreamingJobRunner.get_job_type(),
                input_type="split",
                requires=[],
                required_by_dataset_viewer=True,
                ancestors=[],
                children=[],
                job_runner_version=SplitFirstRowsFromStreamingJobRunner.get_job_runner_version(),
            ),
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
            first_rows_config=first_rows_config,
            assets_directory=assets_directory,
        )

    return _get_job_runner


def test_should_skip_job(
    app_config: AppConfig, get_job_runner: GetJobRunner, first_rows_config: FirstRowsConfig, hub_public_csv: str
) -> None:
    dataset, config, split = get_default_config_split(hub_public_csv)
    job_runner = get_job_runner(dataset, config, split, app_config, first_rows_config, False)
    assert not job_runner.should_skip_job()
    # we add an entry to the cache
    upsert_response(
        kind="/split-names-from-streaming",
        dataset=dataset,
        config=config,
        content={"splits": [{"dataset": dataset, "config": config, "split": split}]},
        http_status=HTTPStatus.OK,
    )
    job_runner.process()
    assert job_runner.should_skip_job()
    job_runner = get_job_runner(dataset, config, split, app_config, first_rows_config, True)
    assert not job_runner.should_skip_job()


def test_compute(
    app_config: AppConfig, get_job_runner: GetJobRunner, first_rows_config: FirstRowsConfig, hub_public_csv: str
) -> None:
    dataset, config, split = get_default_config_split(hub_public_csv)
    job_runner = get_job_runner(dataset, config, split, app_config, first_rows_config, False)
    upsert_response(
        kind="/split-names-from-streaming",
        dataset=dataset,
        config=config,
        content={"splits": [{"dataset": dataset, "config": config, "split": split}]},
        http_status=HTTPStatus.OK,
    )
    assert job_runner.process()
    cached_response = get_response(
        kind=job_runner.processing_step.cache_kind, dataset=dataset, config=config, split=split
    )
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None
    assert cached_response["job_runner_version"] == job_runner.get_job_runner_version()
    assert cached_response["dataset_git_revision"] is not None
    content = cached_response["content"]
    assert content["features"][0]["feature_idx"] == 0
    assert content["features"][0]["name"] == "col_1"
    assert content["features"][0]["type"]["_type"] == "Value"
    assert content["features"][0]["type"]["dtype"] == "int64"  # <---|
    assert content["features"][1]["type"]["dtype"] == "int64"  # <---|- auto-detected by the datasets library
    assert content["features"][2]["type"]["dtype"] == "float64"  # <-|


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner, first_rows_config: FirstRowsConfig) -> None:
    dataset = "doesnotexist"
    dataset, config, split = get_default_config_split(dataset)
    job_runner = get_job_runner(dataset, config, split, app_config, first_rows_config, False)
    assert not job_runner.process()
    with pytest.raises(DoesNotExist):
        get_response(kind=job_runner.processing_step.cache_kind, dataset=dataset, config=config, split=split)


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
        ("does_not_exist_config", False, "ConfigNotFoundError", "DoesNotExist"),
        # should we really test the following cases?
        # The assumption is that the dataset exists and is accessible with the token
        ("does_not_exist_split", False, "SplitNotFoundError", None),
        ("gated", False, "InfoError", "FileNotFoundError"),
        ("private", False, "InfoError", "FileNotFoundError"),
    ],
)
def test_number_rows(
    hub_datasets: HubDatasets,
    get_job_runner: GetJobRunner,
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
    if hasattr(csv, "_patched_for_streaming") and csv._patched_for_streaming:
        csv._patched_for_streaming = False

    dataset = hub_datasets[name]["name"]
    expected_first_rows_response = hub_datasets[name]["first_rows_response"]
    dataset, config, split = get_default_config_split(dataset)
    job_runner = get_job_runner(
        dataset,
        config,
        split,
        app_config if use_token else replace(app_config, common=replace(app_config.common, hf_token=None)),
        first_rows_config,
        False,
    )
    job_runner.get_dataset_git_revision = Mock(return_value="1.0.0")  # type: ignore

    if error_code is None:
        upsert_response(
            kind="/split-names-from-streaming",
            dataset=dataset,
            config=config,
            content={"splits": [{"dataset": dataset, "config": config, "split": split}]},
            http_status=HTTPStatus.OK,
        )
        result = job_runner.compute().content
        assert result == expected_first_rows_response
        return
    elif error_code == "SplitNotFoundError":
        upsert_response(
            kind="/split-names-from-streaming",
            dataset=dataset,
            config=config,
            content={"splits": [{"dataset": dataset, "config": config, "split": "other_split"}]},
            http_status=HTTPStatus.OK,
        )
    elif error_code in {"InfoError", "SplitsNamesError"}:
        upsert_response(
            kind="/split-names-from-streaming",
            dataset=dataset,
            config=config,
            content={"splits": [{"dataset": dataset, "config": config, "split": split}]},
            http_status=HTTPStatus.OK,
        )

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
    "name,rows_max_bytes,columns_max_number,error_code",
    [
        # not-truncated public response is 687 bytes
        ("public", 10, 1_000, "TooBigContentError"),  # too small limit, even with truncation
        ("public", 1_000, 1_000, None),  # not truncated
        ("public", 1_000, 1, "TooManyColumnsError"),  # too small columns limit
        # not-truncated big response is 5_885_989 bytes
        ("big", 10, 1_000, "TooBigContentError"),  # too small limit, even with truncation
        ("big", 1_000, 1_000, None),  # truncated successfully
        ("big", 10_000_000, 1_000, None),  # not truncated
    ],
)
def test_truncation(
    hub_datasets: HubDatasets,
    get_job_runner: GetJobRunner,
    app_config: AppConfig,
    first_rows_config: FirstRowsConfig,
    name: str,
    rows_max_bytes: int,
    columns_max_number: int,
    error_code: str,
) -> None:
    dataset, config, split = get_default_config_split(hub_datasets[name]["name"])
    job_runner = get_job_runner(
        dataset,
        config,
        split,
        replace(app_config, common=replace(app_config.common, hf_token=None)),
        replace(
            first_rows_config,
            max_number=1_000_000,
            min_number=10,
            max_bytes=rows_max_bytes,
            min_cell_bytes=10,
            columns_max_number=columns_max_number,
        ),
        False,
    )

    upsert_response(
        kind="/split-names-from-streaming",
        dataset=dataset,
        config=config,
        content={"splits": [{"dataset": dataset, "config": config, "split": split}]},
        http_status=HTTPStatus.OK,
    )

    job_runner.get_dataset_git_revision = Mock(return_value="1.0.0")  # type: ignore

    if error_code:
        with pytest.raises(CustomError) as error_info:
            job_runner.compute()
        assert error_info.value.code == error_code
    else:
        response = job_runner.compute().content
        assert get_json_size(response) <= rows_max_bytes


@pytest.mark.parametrize(
    "streaming_response_status,dataset_git_revision,error_code,status_code",
    [
        (HTTPStatus.OK, "CURRENT_GIT_REVISION", "ResponseAlreadyComputedError", HTTPStatus.INTERNAL_SERVER_ERROR),
        (HTTPStatus.INTERNAL_SERVER_ERROR, "CURRENT_GIT_REVISION", "ConfigNotFoundError", HTTPStatus.NOT_FOUND),
        (HTTPStatus.OK, "DIFFERENT_GIT_REVISION", "ConfigNotFoundError", HTTPStatus.NOT_FOUND),
    ],
)
def test_response_already_computed(
    app_config: AppConfig,
    first_rows_config: FirstRowsConfig,
    get_job_runner: GetJobRunner,
    streaming_response_status: HTTPStatus,
    dataset_git_revision: str,
    error_code: str,
    status_code: HTTPStatus,
) -> None:
    dataset = "dataset"
    config = "config"
    split = "split"
    current_dataset_git_revision = "CURRENT_GIT_REVISION"
    upsert_response(
        kind="split-first-rows-from-parquet",
        dataset=dataset,
        config=config,
        split=split,
        content={},
        dataset_git_revision=dataset_git_revision,
        job_runner_version=PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_PARQUET_VERSION,
        progress=1.0,
        http_status=streaming_response_status,
    )
    job_runner = get_job_runner(
        dataset,
        config,
        split,
        app_config,
        first_rows_config,
        False,
    )
    job_runner.get_dataset_git_revision = Mock(return_value=current_dataset_git_revision)  # type: ignore
    with pytest.raises(CustomError) as exc_info:
        job_runner.compute()
    assert exc_info.value.status_code == status_code
    assert exc_info.value.code == error_code
