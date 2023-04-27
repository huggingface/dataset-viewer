# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os
from dataclasses import replace
from http import HTTPStatus
from typing import Callable
from unittest.mock import Mock, patch

import pytest
from libcommon.constants import PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_STREAMING_VERSION
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import DoesNotExist, get_response, upsert_response
from libcommon.storage import StrPath
from pyarrow.fs import LocalFileSystem

from worker.config import AppConfig
from worker.job_runners.split.first_rows_from_parquet import (
    SplitFirstRowsFromParquetJobRunner,
)
from worker.utils import get_json_size

from ...fixtures.hub import get_default_config_split

GetJobRunner = Callable[[str, str, str, AppConfig, bool], SplitFirstRowsFromParquetJobRunner]


@pytest.fixture
def get_job_runner(
    assets_directory: StrPath,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        split: str,
        app_config: AppConfig,
        force: bool = False,
    ) -> SplitFirstRowsFromParquetJobRunner:
        return SplitFirstRowsFromParquetJobRunner(
            job_info={
                "type": SplitFirstRowsFromParquetJobRunner.get_job_type(),
                "dataset": dataset,
                "config": config,
                "split": split,
                "job_id": "job_id",
                "force": force,
                "priority": Priority.NORMAL,
            },
            app_config=app_config,
            processing_step=ProcessingStep(
                name=SplitFirstRowsFromParquetJobRunner.get_job_type(),
                input_type="split",
                required_by_dataset_viewer=True,
                job_runner_version=SplitFirstRowsFromParquetJobRunner.get_job_runner_version(),
            ),
            assets_directory=assets_directory,
        )

    return _get_job_runner


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "doesnotexist"
    dataset, config, split = get_default_config_split(dataset)
    job_runner = get_job_runner(dataset, config, split, app_config, False)
    assert not job_runner.process()
    with pytest.raises(DoesNotExist):
        get_response(kind=job_runner.processing_step.cache_kind, dataset=dataset, config=config, split=split)


@pytest.mark.parametrize(
    "rows_max_bytes,columns_max_number,error_code",
    [
        (0, 10, "TooBigContentError"),  # too small limit, even with truncation
        (1_000, 1, "TooManyColumnsError"),  # too small columns limit
        (1_000, 10, None),
    ],
)
def test_compute(
    get_job_runner: GetJobRunner,
    app_config: AppConfig,
    rows_max_bytes: int,
    columns_max_number: int,
    error_code: str,
) -> None:
    dataset, config, split = "dataset", "config", "split"
    upsert_response(
        kind="config-parquet",
        dataset=dataset,
        config=config,
        content={
            "parquet_files": [
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "filename": f"{dataset}-{split}.parquet",
                    "size": 1000,
                }
            ]
        },
        http_status=HTTPStatus.OK,
    )

    with patch("worker.job_runners.split.first_rows_from_parquet.get_parquet_fs") as mock_read:
        initial_location = os.getcwd()
        os.chdir("tests/job_runners/split")
        # TODO:  Make localsystem by relative path
        fs = LocalFileSystem()
        mock_read.return_value = fs
        # ^ Mocking file system with local file
        job_runner = get_job_runner(
            dataset,
            config,
            split,
            replace(
                app_config,
                common=replace(app_config.common, hf_token=None),
                first_rows=replace(
                    app_config.first_rows,
                    max_number=1_000_000,
                    min_number=10,
                    max_bytes=rows_max_bytes,
                    min_cell_bytes=10,
                    columns_max_number=columns_max_number,
                ),
            ),
            False,
        )

        job_runner.get_dataset_git_revision = Mock(return_value="1.0.0")  # type: ignore
        if error_code:
            with pytest.raises(CustomError) as error_info:
                job_runner.compute()
            assert error_info.value.code == error_code
        else:
            response = job_runner.compute().content
            assert get_json_size(response) <= rows_max_bytes
            assert response
            assert response["rows"]
            assert response["features"]
            assert len(response["rows"]) == 3  # testing file has 3 rows see config/dataset-split.parquet file
            assert len(response["features"]) == 2  # testing file has 2 columns see config/dataset-split.parquet file
            assert response["features"][0]["feature_idx"] == 0
            assert response["features"][0]["name"] == "col1"
            assert response["features"][0]["type"]["_type"] == "Value"
            assert response["features"][0]["type"]["dtype"] == "int32"
            assert response["features"][1]["feature_idx"] == 1
            assert response["features"][1]["name"] == "col2"
            assert response["features"][1]["type"]["_type"] == "Value"
            assert response["features"][1]["type"]["dtype"] == "string"
            assert response["rows"][0]["row_idx"] == 0
            assert response["rows"][0]["truncated_cells"] == []
            assert response["rows"][0]["row"] == {"col1": 1, "col2": "a"}
            assert response["rows"][1]["row_idx"] == 1
            assert response["rows"][1]["truncated_cells"] == []
            assert response["rows"][1]["row"] == {"col1": 2, "col2": "b"}
            assert response["rows"][2]["row_idx"] == 2
            assert response["rows"][2]["truncated_cells"] == []
            assert response["rows"][2]["row"] == {"col1": 3, "col2": "c"}
        os.chdir(initial_location)


@pytest.mark.parametrize(
    "streaming_response_status,dataset_git_revision,error_code,status_code",
    [
        (HTTPStatus.OK, "CURRENT_GIT_REVISION", "ResponseAlreadyComputedError", HTTPStatus.INTERNAL_SERVER_ERROR),
        (HTTPStatus.INTERNAL_SERVER_ERROR, "CURRENT_GIT_REVISION", "CachedResponseNotFound", HTTPStatus.NOT_FOUND),
        (HTTPStatus.OK, "DIFFERENT_GIT_REVISION", "CachedResponseNotFound", HTTPStatus.NOT_FOUND),
    ],
)
def test_response_already_computed(
    app_config: AppConfig,
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
        kind="split-first-rows-from-streaming",
        dataset=dataset,
        config=config,
        split=split,
        content={},
        dataset_git_revision=dataset_git_revision,
        job_runner_version=PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_STREAMING_VERSION,
        progress=1.0,
        http_status=streaming_response_status,
    )
    job_runner = get_job_runner(
        dataset,
        config,
        split,
        app_config,
        False,
    )
    job_runner.get_dataset_git_revision = Mock(return_value=current_dataset_git_revision)  # type: ignore
    with pytest.raises(CustomError) as exc_info:
        job_runner.compute()
    assert exc_info.value.status_code == status_code
    assert exc_info.value.code == error_code
