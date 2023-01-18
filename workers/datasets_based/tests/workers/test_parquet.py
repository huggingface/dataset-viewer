# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any

import pytest
from libcommon.simple_cache import _clean_cache_database, upsert_response

from datasets_based.config import AppConfig
from datasets_based.workers.parquet import (
    DatasetNotFoundError,
    ParquetWorker,
    PreviousStepFormatError,
    PreviousStepStatusError,
)


@pytest.fixture(autouse=True)
def clean_mongo_database(app_config: AppConfig) -> None:
    _clean_cache_database()


def get_worker(dataset: str, app_config: AppConfig, force: bool = False) -> ParquetWorker:
    return ParquetWorker(
        job_info={
            "type": ParquetWorker.get_job_type(),
            "dataset": dataset,
            "config": None,
            "split": None,
            "job_id": "job_id",
            "force": force,
        },
        app_config=app_config,
    )


@pytest.mark.parametrize(
    "dataset,upstream_status,upstream_content,expected_error_code,expected_status,expected_content",
    [
        (
            "ok",
            HTTPStatus.OK,
            {"parquet_files": [{"key": "value"}], "dataset_info": {"key": "value"}},
            None,
            HTTPStatus.OK,
            {"parquet_files": [{"key": "value"}]},
        ),
        (
            "status_error",
            HTTPStatus.NOT_FOUND,
            {"error": "error"},
            PreviousStepStatusError.__name__,
            HTTPStatus.INTERNAL_SERVER_ERROR,
            None,
        ),
        (
            "format_error",
            HTTPStatus.OK,
            {"not_parquet_files": "wrong_format"},
            PreviousStepFormatError.__name__,
            HTTPStatus.INTERNAL_SERVER_ERROR,
            None,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    dataset: str,
    upstream_status: HTTPStatus,
    upstream_content: Any,
    expected_error_code: str,
    expected_status: HTTPStatus,
    expected_content: Any,
) -> None:
    upsert_response(
        kind="/parquet-and-dataset-info", dataset=dataset, content=upstream_content, http_status=upstream_status
    )
    worker = get_worker(dataset=dataset, app_config=app_config)
    if expected_status == HTTPStatus.OK:
        assert worker.compute() == expected_content
    else:
        with pytest.raises(Exception) as e:
            worker.compute()
        assert e.type.__name__ == expected_error_code


def test_doesnotexist(app_config: AppConfig) -> None:
    dataset = "doesnotexist"
    worker = get_worker(dataset=dataset, app_config=app_config)
    with pytest.raises(DatasetNotFoundError):
        worker.compute()
