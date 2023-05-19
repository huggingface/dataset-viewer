# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Callable

import pytest
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedArtifactError, upsert_response
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.job_runners.split.opt_in_out_urls_count import (
    SplitOptInOutUrlsCountJobRunner,
)


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, str, str, AppConfig], SplitOptInOutUrlsCountJobRunner]


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        split: str,
        app_config: AppConfig,
    ) -> SplitOptInOutUrlsCountJobRunner:
        processing_step_name = SplitOptInOutUrlsCountJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            {
                "dataset-level": {"input_type": "dataset"},
                "config-level": {"input_type": "dataset", "triggered_by": "dataset-level"},
                processing_step_name: {
                    "input_type": "split",
                    "job_runner_version": SplitOptInOutUrlsCountJobRunner.get_job_runner_version(),
                    "triggered_by": "config-level",
                },
            }
        )
        return SplitOptInOutUrlsCountJobRunner(
            job_info={
                "type": SplitOptInOutUrlsCountJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": config,
                    "split": split,
                    "partition_start": None,
                    "partition_end": None,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,config,split,upstream_status,upstream_content,expected_error_code,expected_content,should_raise",
    [
        (
            "dataset_ok",
            "config_ok",
            "split_ok",
            HTTPStatus.OK,
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
                "num_opt_out_urls": 1,
                "num_opt_in_urls": 1,
                "num_urls": 4,
                "full_scan": True,
            },
            None,
            {
                "has_urls_columns": True,
                "num_scanned_rows": 4,
                "urls_columns": ["col"],
                "num_opt_out_urls": 1,
                "num_opt_in_urls": 1,
                "num_urls": 4,
                "full_scan": True,
            },
            False,
        ),
        (
            "dataset_previous_step_error",
            "config_previous_step_error",
            "split_previous_step_error",
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {},
            "CachedArtifactError",
            None,
            True,
        ),
        (
            "dataset_format_error",
            "config_format_error",
            "split_format_error",
            HTTPStatus.OK,
            {"wrong_format": None},
            "PreviousStepFormatError",
            None,
            True,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    config: str,
    split: str,
    upstream_status: HTTPStatus,
    upstream_content: Any,
    expected_error_code: str,
    expected_content: Any,
    should_raise: bool,
) -> None:
    upsert_response(
        kind="split-opt-in-out-urls-scan",
        dataset=dataset,
        config=config,
        split=split,
        content=upstream_content,
        http_status=upstream_status,
    )
    job_runner = get_job_runner(dataset, config, split, app_config)
    if should_raise:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.typename == expected_error_code
    else:
        assert job_runner.compute().content == expected_content


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = config = split = "doesnotexist"
    job_runner = get_job_runner(dataset, config, split, app_config)
    with pytest.raises(CachedArtifactError):
        job_runner.compute()
