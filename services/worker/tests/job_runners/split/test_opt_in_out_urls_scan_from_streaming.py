# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from asyncio import Semaphore
from dataclasses import replace
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, List, Mapping
from unittest.mock import patch

import pandas as pd
import pytest
from aiohttp import ClientSession
from aiolimiter import AsyncLimiter
from libcommon.constants import PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import get_response, upsert_response
from libcommon.storage import StrPath

from worker.config import AppConfig
from worker.job_runners.split.opt_in_out_urls_scan_from_streaming import (
    ExternalServerError,
    SplitOptInOutUrlsScanJobRunner,
    check_spawning,
)
from worker.resources import LibrariesResource

from ...constants import CI_SPAWNING_TOKEN
from ...fixtures.hub import HubDatasets, get_default_config_split

GetJobRunner = Callable[[str, str, str, AppConfig, bool], SplitOptInOutUrlsScanJobRunner]


async def mock_check_spawning(
    image_urls: List[str], session: ClientSession, semaphore: Semaphore, limiter: AsyncLimiter, url: str
) -> Any:
    return {"urls": [{"url": url, "optIn": "optIn" in url, "optOut": "optOut" in url} for url in image_urls]}


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
            assets_directory=assets_directory,
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
        {"row_idx": 0, "row": {"col": "http://testurl.test/test_image-optOut.jpg"}, "truncated_cells": []},
        {"row_idx": 1, "row": {"col": "http://testurl.test/test_image2.jpg"}, "truncated_cells": []},
        {"row_idx": 2, "row": {"col": "other"}, "truncated_cells": []},
        {"row_idx": 3, "row": {"col": "http://testurl.test/test_image3-optIn.jpg"}, "truncated_cells": []},
    ],
}

OPT_IN_OUT_DATAFRAME = {
    "url": ["http://testurl.test/test_image-optOut.jpg", "http://testurl.test/test_image3-optIn.jpg"],
    "row_idx": [0, 3],
    "feature_name": ["col", "col"],
    "is_opt_in": [False, True],
    "is_opt_out": [True, False],
}


@pytest.mark.parametrize(
    "name,upstream_content,expected_content,expected_dataframe",
    [
        (
            "public",
            FIRST_ROWS_WITHOUT_OPT_IN_OUT_URLS,
            {
                "has_urls_columns": False,
                "num_scanned_rows": 0,
                "urls_columns": [],
                "num_opt_out_urls": 0,
                "num_opt_in_urls": 0,
                "num_urls": 0,
            },
            None,
        ),
        (
            "spawning_opt_in_out",
            FIRST_ROWS_WITH_OPT_IN_OUT_URLS,
            {
                "has_urls_columns": True,
                "num_scanned_rows": 4,
                "urls_columns": ["col"],
                "num_opt_out_urls": 1,
                "num_opt_in_urls": 1,
                "num_urls": 4,
            },
            pd.DataFrame(OPT_IN_OUT_DATAFRAME),
        ),
    ],
)
def test_compute(
    assets_directory: StrPath,
    hub_datasets: HubDatasets,
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    name: str,
    upstream_content: Mapping[str, Any],
    expected_content: Mapping[str, Any],
    expected_dataframe: pd.DataFrame,
) -> None:
    dataset, config, split = get_default_config_split(hub_datasets[name]["name"])
    job_runner = get_job_runner(dataset, config, split, app_config, False)
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
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None

    content = cached_response["content"]
    assert content
    assert content["has_urls_columns"] == expected_content["has_urls_columns"]
    assert content["num_scanned_rows"] == expected_content["num_scanned_rows"]
    assert content["urls_columns"] == expected_content["urls_columns"]
    assert content["num_opt_out_urls"] == expected_content["num_opt_out_urls"]
    assert content["num_opt_in_urls"] == expected_content["num_opt_in_urls"]
    assert content["num_urls"] == expected_content["num_urls"]
    if expected_content["has_urls_columns"]:
        assert content["urls_file"] is not None and isinstance(content["urls_file"], dict)
        assert content["urls_file"]["src"] is not None
        assert content["urls_file"]["type"] == "text/csv"
        file_location = Path(assets_directory).resolve() / dataset / "--" / config / split / "opt_in_out.csv"
        data = pd.read_csv(file_location)
        assert data is not None
        assert data.equals(expected_dataframe)
    else:
        assert content["urls_file"] is None


@pytest.mark.parametrize(
    "dataset,columns_max_number,upstream_content,upstream_status,error_code,status_code",
    [
        ("doesnotexist", 10, {}, HTTPStatus.OK, "CachedResponseNotFound", HTTPStatus.NOT_FOUND),
        ("wrong_format", 10, {}, HTTPStatus.OK, "PreviousStepFormatError", HTTPStatus.INTERNAL_SERVER_ERROR),
        (
            "upstream_failed",
            10,
            {},
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "PreviousStepError",
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
            0,
            FIRST_ROWS_WITH_OPT_IN_OUT_URLS,
            HTTPStatus.OK,
            "TooManyColumnsError",
            HTTPStatus.INTERNAL_SERVER_ERROR,
        ),
    ],
)
def test_compute_failed(
    app_config: AppConfig,
    hub_datasets: HubDatasets,
    get_job_runner: GetJobRunner,
    dataset: str,
    columns_max_number: int,
    upstream_content: Mapping[str, Any],
    upstream_status: HTTPStatus,
    error_code: str,
    status_code: HTTPStatus,
) -> None:
    if dataset == "too_many_columns":
        dataset = hub_datasets["spawning_opt_in_out"]["name"]
    dataset, config, split = get_default_config_split(dataset)
    job_runner = get_job_runner(
        dataset,
        config,
        split,
        replace(app_config, urls_scan=replace(app_config.urls_scan, columns_max_number=columns_max_number)),
        False,
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
    hub_public_spawning_opt_in_out: str,
) -> None:
    dataset, config, split = get_default_config_split(hub_public_spawning_opt_in_out)
    job_runner = get_job_runner(
        dataset,
        config,
        split,
        replace(app_config, urls_scan=replace(app_config.urls_scan, spawning_url="wrong_url")),
        False,
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
    with patch("worker.job_runners.split.opt_in_out_urls_scan_from_streaming.RETRY_TIMES", [1]):
        with patch("worker.job_runners.split.opt_in_out_urls_scan_from_streaming.TIMEOUT", 1):
            with pytest.raises(ExternalServerError):
                job_runner.compute()


@pytest.mark.asyncio
async def test_real_check_spawning_response(app_config: AppConfig) -> None:
    semaphore = Semaphore(value=10)
    limiter = AsyncLimiter(10, time_period=1)

    headers = {"Authorization": f"API {CI_SPAWNING_TOKEN}"}
    async with ClientSession(headers=headers) as session:
        image_url = "http://testurl.test/test_image.jpg"
        image_urls = [image_url]
        spawning_url = app_config.urls_scan.spawning_url
        spawning_response = await check_spawning(image_urls, session, semaphore, limiter, spawning_url)
        assert spawning_response and isinstance(spawning_response, dict)
        assert spawning_response["urls"] and isinstance(spawning_response["urls"], list)
        assert len(spawning_response["urls"]) == 2  # the API requires >1 urls
        first_url = spawning_response["urls"][0]
        assert first_url and isinstance(first_url, dict)
        assert first_url["url"] and isinstance(first_url["url"], str)
        assert first_url["url"] == image_url
