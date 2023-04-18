# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import replace
from http import HTTPStatus
from typing import Callable

import pytest
from libcommon.constants import PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import get_response, upsert_response

from worker.config import AppConfig, OptInOutUrlsScanConfig
from worker.job_runner import ConfigNotFoundError
from worker.job_runners.split.opt_in_out_urls_scan_from_streaming import (
    ExternalServerError,
    InfoError,
    PreviousStepFormatError,
    PreviousStepStatusError,
    SplitOptInOutUrlsScanJobRunner,
    TooManyColumnsError,
)
from worker.resources import LibrariesResource

from ...fixtures.hub import get_default_config_split

GetJobRunner = Callable[[str, str, str, AppConfig, OptInOutUrlsScanConfig, bool], SplitOptInOutUrlsScanJobRunner]


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        split: str,
        app_config: AppConfig,
        urls_scan_config: OptInOutUrlsScanConfig,
        force: bool = False,
    ) -> SplitOptInOutUrlsScanJobRunner:
        return SplitOptInOutUrlsScanJobRunner(
            job_info={
                "type": SplitOptInOutUrlsScanJobRunner.get_job_type(),
                "dataset": dataset,
                "config": config,
                "split": split,
                "job_id": "job_id",
                "force": force,
                "priority": Priority.NORMAL,
            },
            app_config=app_config,
            processing_step=ProcessingStep(
                name=SplitOptInOutUrlsScanJobRunner.get_job_type(),
                input_type="split",
                requires=[],
                required_by_dataset_viewer=True,
                ancestors=[],
                children=[],
                parents=[],
                job_runner_version=SplitOptInOutUrlsScanJobRunner.get_job_runner_version(),
            ),
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
            urls_scan_config=urls_scan_config,
        )

    return _get_job_runner


def test_upstream_content_does_not_exist(
    app_config: AppConfig, get_job_runner: GetJobRunner, urls_scan_config: OptInOutUrlsScanConfig
) -> None:
    dataset = "doesnotexist"
    dataset, config, split = get_default_config_split(dataset)
    job_runner = get_job_runner(dataset, config, split, app_config, urls_scan_config, False)
    with pytest.raises(ConfigNotFoundError):
        job_runner.compute()


def test_upstream_content_wrong_format(
    app_config: AppConfig, get_job_runner: GetJobRunner, urls_scan_config: OptInOutUrlsScanConfig
) -> None:
    dataset = "doesnotexist"
    dataset, config, split = get_default_config_split(dataset)
    job_runner = get_job_runner(dataset, config, split, app_config, urls_scan_config, False)
    upsert_response(
        kind="split-first-rows-from-streaming",
        dataset=dataset,
        config=config,
        split=split,
        content={},
        dataset_git_revision="dataset_git_revision",
        job_runner_version=PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION,
        progress=1.0,
        http_status=HTTPStatus.OK,
    )
    with pytest.raises(PreviousStepFormatError):
        job_runner.compute()


def test_upstream_content_failed(
    app_config: AppConfig, get_job_runner: GetJobRunner, urls_scan_config: OptInOutUrlsScanConfig
) -> None:
    dataset = "doesnotexist"
    dataset, config, split = get_default_config_split(dataset)
    job_runner = get_job_runner(dataset, config, split, app_config, urls_scan_config, False)
    upsert_response(
        kind="split-first-rows-from-streaming",
        dataset=dataset,
        config=config,
        split=split,
        content={},
        dataset_git_revision="dataset_git_revision",
        job_runner_version=PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION,
        progress=1.0,
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    )
    with pytest.raises(PreviousStepStatusError):
        job_runner.compute()


def test_compute_too_many_columns_error(
    app_config: AppConfig, get_job_runner: GetJobRunner, urls_scan_config: OptInOutUrlsScanConfig
) -> None:
    dataset = "doesnotexist"
    dataset, config, split = get_default_config_split(dataset)
    job_runner = get_job_runner(
        dataset, config, split, app_config, replace(urls_scan_config, columns_max_number=1), False
    )
    upsert_response(
        kind="split-first-rows-from-streaming",
        dataset=dataset,
        config=config,
        split=split,
        content={
            "features": [
                {
                    "feature_idx": 0,
                    "name": "feature1",
                    "type": {
                        "dtype": "string",
                        "_type": "Value",
                    },
                },
                {
                    "feature_idx": 1,
                    "name": "feature2",
                    "type": {
                        "dtype": "string",
                        "_type": "Value",
                    },
                },
            ],
            "rows": [],
        },
        dataset_git_revision="dataset_git_revision",
        job_runner_version=PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION,
        progress=1.0,
        http_status=HTTPStatus.OK,
    )
    with pytest.raises(TooManyColumnsError):
        job_runner.compute()


def test_compute_info_error(
    app_config: AppConfig, get_job_runner: GetJobRunner, urls_scan_config: OptInOutUrlsScanConfig
) -> None:
    dataset, config, split = "doesnotexist", "config", "split"
    job_runner = get_job_runner(dataset, config, split, app_config, urls_scan_config, False)
    upsert_response(
        kind="split-first-rows-from-streaming",
        dataset=dataset,
        config=config,
        split=split,
        content={
            "features": [
                {
                    "feature_idx": 0,
                    "name": "col1",
                    "type": {
                        "dtype": "int64",
                        "_type": "Value",
                    },
                },
                {
                    "feature_idx": 1,
                    "name": "col2",
                    "type": {
                        "dtype": "int64",
                        "_type": "Value",
                    },
                },
                {
                    "feature_idx": 2,
                    "name": "col3",
                    "type": {
                        "dtype": "float64",
                        "_type": "Value",
                    },
                },
            ],
            "rows": [],
        },
        dataset_git_revision="dataset_git_revision",
        job_runner_version=PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION,
        progress=1.0,
        http_status=HTTPStatus.OK,
    )
    with pytest.raises(InfoError):
        job_runner.compute()


def test_compute_no_opt_in_out_urls(
    app_config: AppConfig, get_job_runner: GetJobRunner, urls_scan_config: OptInOutUrlsScanConfig, hub_public_csv: str
) -> None:
    dataset, config, split = get_default_config_split(hub_public_csv)
    job_runner = get_job_runner(dataset, config, split, app_config, urls_scan_config, False)
    upsert_response(
        kind="split-first-rows-from-streaming",
        dataset=dataset,
        config=config,
        split=split,
        content={
            "features": [
                {
                    "feature_idx": 0,
                    "name": "col1",
                    "type": {
                        "dtype": "int64",
                        "_type": "Value",
                    },
                },
                {
                    "feature_idx": 1,
                    "name": "col2",
                    "type": {
                        "dtype": "int64",
                        "_type": "Value",
                    },
                },
                {
                    "feature_idx": 2,
                    "name": "col3",
                    "type": {
                        "dtype": "float64",
                        "_type": "Value",
                    },
                },
            ],
            "rows": [],
        },
        dataset_git_revision="dataset_git_revision",
        job_runner_version=PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION,
        progress=1.0,
        http_status=HTTPStatus.OK,
    )

    assert job_runner.process()
    cached_response = get_response(
        kind=job_runner.processing_step.cache_kind, dataset=dataset, config=config, split=split
    )
    assert cached_response
    content = cached_response["content"]
    assert content
    assert not content["has_urls_columns"]
    assert content["urls_columns"] == []
    assert content["opt_in_urls"] == []
    assert content["opt_out_urls"] == []
    assert content["opt_in_urls_indices"] == []
    assert content["opt_out_urls_indices"] == []
    assert content["num_scanned_rows"] == 0
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None


def test_compute_with_opt_in_out_urls(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    urls_scan_config: OptInOutUrlsScanConfig,
    hub_public_spawning_opt_in_out: str,
) -> None:
    dataset, config, split = get_default_config_split(hub_public_spawning_opt_in_out)
    job_runner = get_job_runner(dataset, config, split, app_config, urls_scan_config, False)
    upsert_response(
        kind="split-first-rows-from-streaming",
        dataset=dataset,
        config=config,
        split=split,
        content={
            "features": [
                {
                    "feature_idx": 0,
                    "name": "col",
                    "type": {
                        "dtype": "string",
                        "_type": "Value",
                    },
                }
            ],
            "rows": [
                {"row_idx": 0, "row": {"col": "http://testurl.test/test_image.jpg"}, "truncated_cells": []},
                {"row_idx": 1, "row": {"col": "http://testurl.test/test_image2.jpg"}, "truncated_cells": []},
                {"row_idx": 2, "row": {"col": "other"}, "truncated_cells": []},
            ],
        },
        dataset_git_revision="dataset_git_revision",
        job_runner_version=PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION,
        progress=1.0,
        http_status=HTTPStatus.OK,
    )

    assert job_runner.process()
    cached_response = get_response(
        kind=job_runner.processing_step.cache_kind, dataset=dataset, config=config, split=split
    )
    assert cached_response
    content = cached_response["content"]
    assert content
    assert content["has_urls_columns"]
    assert content["urls_columns"] == ["col"]
    assert content["opt_in_urls"] == []
    assert content["opt_out_urls"] == []
    assert content["opt_in_urls_indices"] == []
    assert content["opt_out_urls_indices"] == []
    assert content["num_scanned_rows"] == 3
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None


def test_compute_error_from_spawning(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    urls_scan_config: OptInOutUrlsScanConfig,
    hub_public_spawning_opt_in_out: str,
) -> None:
    dataset, config, split = get_default_config_split(hub_public_spawning_opt_in_out)
    job_runner = get_job_runner(dataset, config, split, app_config, replace(urls_scan_config, url="wrong_url"), False)
    upsert_response(
        kind="split-first-rows-from-streaming",
        dataset=dataset,
        config=config,
        split=split,
        content={
            "features": [
                {
                    "feature_idx": 0,
                    "name": "col",
                    "type": {
                        "dtype": "string",
                        "_type": "Value",
                    },
                }
            ],
            "rows": [
                {"row_idx": 0, "row": {"col": "http://testurl.test/test_image.jpg"}, "truncated_cells": []},
                {"row_idx": 1, "row": {"col": "http://testurl.test/test_image2.jpg"}, "truncated_cells": []},
                {"row_idx": 2, "row": {"col": "other"}, "truncated_cells": []},
            ],
        },
        dataset_git_revision="dataset_git_revision",
        job_runner_version=PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION,
        progress=1.0,
        http_status=HTTPStatus.OK,
    )
    with pytest.raises(ExternalServerError):
        job_runner.compute()
