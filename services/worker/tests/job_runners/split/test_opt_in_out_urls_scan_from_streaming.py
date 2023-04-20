# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from asyncio import Semaphore
from dataclasses import replace
from http import HTTPStatus
from typing import Any, Callable, List, Mapping
from unittest.mock import patch

import pytest
from aiohttp import ClientSession
from aiolimiter import AsyncLimiter
from libcommon.constants import PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import get_response, upsert_response

from worker.config import AppConfig, OptInOutUrlsScanConfig
from worker.job_runners.split.opt_in_out_urls_scan_from_streaming import (
    ExternalServerError,
    SplitOptInOutUrlsScanJobRunner,
)
from worker.resources import LibrariesResource

from ...fixtures.hub import HubDatasets, get_default_config_split

GetJobRunner = Callable[[str, str, str, AppConfig, OptInOutUrlsScanConfig, bool], SplitOptInOutUrlsScanJobRunner]


async def mock_check_spawning(
    image_urls: List[str], session: ClientSession, semaphore: Semaphore, limiter: AsyncLimiter, url: str
) -> Any:
    return {"urls": [{"url": url, "optIn": "optIn" in url, "optOut": "optOut" in url} for url in image_urls]}


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


FIRST_ROWS_WITHOUT_OPT_IN_OUT_URLS = {
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
}


FIRST_ROWS_WITH_OPT_IN_OUT_URLS = {
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
}


@pytest.mark.parametrize(
    "name,upstream_content,expected_content",
    [
        (
            "public",
            FIRST_ROWS_WITHOUT_OPT_IN_OUT_URLS,
            {
                "has_urls_columns": False,
                "num_scanned_rows": 0,
                "opt_in_urls": [],
                "opt_out_urls": [],
                "urls_columns": [],
            },
        ),
        (
            "spawning_opt_in_out",
            FIRST_ROWS_WITH_OPT_IN_OUT_URLS,
            {
                "has_urls_columns": True,
                "num_scanned_rows": 4,
                "opt_in_urls": [
                    {"url": "http://testurl.test/test_image3-optIn.jpg", "row_idx": 3, "column_name": "col"}
                ],
                "opt_out_urls": [
                    {"url": "http://testurl.test/test_image-optOut.jpg", "row_idx": 0, "column_name": "col"}
                ],
                "urls_columns": ["col"],
            },
        ),
    ],
)
def test_compute(
    hub_datasets: HubDatasets,
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    urls_scan_config: OptInOutUrlsScanConfig,
    name: str,
    upstream_content: Mapping[str, Any],
    expected_content: Mapping[str, Any],
) -> None:
    dataset, config, split = get_default_config_split(hub_datasets[name]["name"])
    job_runner = get_job_runner(dataset, config, split, app_config, urls_scan_config, False)
    upsert_response(
        kind="split-first-rows-from-streaming",
        dataset=dataset,
        config=config,
        split=split,
        content=upstream_content,
        dataset_git_revision="dataset_git_revision",
        job_runner_version=PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION,
        progress=1.0,
        http_status=HTTPStatus.OK,
    )
    with patch("worker.job_runners.split.opt_in_out_urls_scan_from_streaming.check_spawning", mock_check_spawning):
        assert job_runner.process()
    cached_response = get_response(
        kind=job_runner.processing_step.cache_kind, dataset=dataset, config=config, split=split
    )
    assert cached_response
    assert cached_response["content"] == expected_content
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None


@pytest.mark.parametrize(
    "dataset,columns_max_number,upstream_content,upstream_status,error_code,status_code",
    [
        ("doesnotexist", 10, {}, HTTPStatus.OK, "ConfigNotFoundError", HTTPStatus.NOT_FOUND),
        ("wrong_format", 10, {}, HTTPStatus.OK, "PreviousStepFormatError", HTTPStatus.INTERNAL_SERVER_ERROR),
        (
            "upstream_failed",
            10,
            {},
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "PreviousStepStatusError",
            HTTPStatus.INTERNAL_SERVER_ERROR,
        ),
        (
            "info_error",
            10,
            FIRST_ROWS_WITHOUT_OPT_IN_OUT_URLS,
            HTTPStatus.OK,
            "InfoError",
            HTTPStatus.INTERNAL_SERVER_ERROR,
        ),
        (
            "too_many_columns",
            1,
            FIRST_ROWS_WITHOUT_OPT_IN_OUT_URLS,
            HTTPStatus.OK,
            "TooManyColumnsError",
            HTTPStatus.INTERNAL_SERVER_ERROR,
        ),
    ],
)
def test_compute_failed(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    urls_scan_config: OptInOutUrlsScanConfig,
    dataset: str,
    columns_max_number: int,
    upstream_content: Mapping[str, Any],
    upstream_status: HTTPStatus,
    error_code: str,
    status_code: HTTPStatus,
) -> None:
    dataset, config, split = get_default_config_split(dataset)
    job_runner = get_job_runner(
        dataset, config, split, app_config, replace(urls_scan_config, columns_max_number=columns_max_number), False
    )
    if dataset != "doesnotexist":
        upsert_response(
            kind="split-first-rows-from-streaming",
            dataset=dataset,
            config=config,
            split=split,
            content=upstream_content,
            dataset_git_revision="dataset_git_revision",
            job_runner_version=PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION,
            progress=1.0,
            http_status=upstream_status,
        )
    with pytest.raises(CustomError) as exc_info:
        job_runner.compute()
    assert exc_info.value.status_code == status_code
    assert exc_info.value.code == error_code


def test_compute_error_from_spawning(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    urls_scan_config: OptInOutUrlsScanConfig,
    hub_public_spawning_opt_in_out: str,
) -> None:
    dataset, config, split = get_default_config_split(hub_public_spawning_opt_in_out)
    job_runner = get_job_runner(
        dataset, config, split, app_config, replace(urls_scan_config, spawning_url="wrong_url"), False
    )
    upsert_response(
        kind="split-first-rows-from-streaming",
        dataset=dataset,
        config=config,
        split=split,
        content=FIRST_ROWS_WITH_OPT_IN_OUT_URLS,
        dataset_git_revision="dataset_git_revision",
        job_runner_version=PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION,
        progress=1.0,
        http_status=HTTPStatus.OK,
    )
    with pytest.raises(ExternalServerError):
        job_runner.compute()
