# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any

import pytest
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
from libcommon.simple_cache import upsert_response

from datasets_based.config import AppConfig
from datasets_based.workers.parquet import (
    DatasetNotFoundError,
    ParquetWorker,
    PreviousStepFormatError,
    PreviousStepStatusError,
)


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


def get_worker(dataset: str, app_config: AppConfig, force: bool = False) -> ParquetWorker:
    return ParquetWorker(
        job_info={
            "type": ParquetWorker.get_job_type(),
            "dataset": dataset,
            "config": None,
            "split": None,
            "job_id": "job_id",
            "force": force,
            "priority": Priority.NORMAL,
        },
        common_config=app_config.common,
        processing_step=ProcessingStep(
            endpoint=ParquetWorker.get_job_type(),
            input_type="dataset",
            requires=None,
            required_by_dataset_viewer=False,
            parent=None,
            ancestors=[],
            children=[],
        ),
    )


@pytest.mark.parametrize(
    "dataset,upstream_status,upstream_content,expected_error_code,expected_content,should_raise",
    [
        (
            "ok",
            HTTPStatus.OK,
            {"parquet_files": [{"key": "value"}], "dataset_info": {"key": "value"}},
            None,
            {"parquet_files": [{"key": "value"}]},
            False,
        ),
        ("status_error", HTTPStatus.NOT_FOUND, {"error": "error"}, PreviousStepStatusError.__name__, None, True),
        (
            "format_error",
            HTTPStatus.OK,
            {"not_parquet_files": "wrong_format"},
            PreviousStepFormatError.__name__,
            None,
            True,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    dataset: str,
    upstream_status: HTTPStatus,
    upstream_content: Any,
    expected_error_code: str,
    expected_content: Any,
    should_raise: bool,
) -> None:
    upsert_response(
        kind="/parquet-and-dataset-info", dataset=dataset, content=upstream_content, http_status=upstream_status
    )
    worker = get_worker(dataset=dataset, app_config=app_config)
    if should_raise:
        with pytest.raises(Exception) as e:
            worker.compute()
        assert e.type.__name__ == expected_error_code
    else:
        assert worker.compute() == expected_content


def test_doesnotexist(app_config: AppConfig) -> None:
    dataset = "doesnotexist"
    worker = get_worker(dataset=dataset, app_config=app_config)
    with pytest.raises(DatasetNotFoundError):
        worker.compute()
