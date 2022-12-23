# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus

import pytest
import requests
from libcommon.simple_cache import get_response, upsert_response

from parquet_based.config import AppConfig
from parquet_based.workers.size import SizeWorker


def get_worker(
    dataset: str,
    app_config: AppConfig,
    force: bool = False,
) -> SizeWorker:
    return SizeWorker(
        job_info={
            "type": SizeWorker.get_job_type(),
            "dataset": dataset,
            "config": None,
            "split": None,
            "job_id": "job_id",
            "force": force,
        },
        app_config=app_config,
    )


@pytest.fixture
def fill_parquet_cache(app_config: AppConfig) -> None:
    dataset = "glue"
    response = requests.get(f"https://datasets-server.huggingface.co/parquet?dataset={dataset}")
    assert response.status_code == HTTPStatus.OK
    content = response.json()
    assert len(content["parquet_files"]) == 34
    upsert_response(
        kind="/parquet",
        dataset=dataset,
        config=None,
        split=None,
        content=content,
        error_code=None,
        http_status=HTTPStatus.OK,
        details=None,
        worker_version="0.0.0",
        dataset_git_revision="i don't know",
    )


@pytest.mark.real_dataset
@pytest.mark.parametrize("dataset", ["glue"])
def test_compute(fill_parquet_cache, app_config: AppConfig, dataset: str) -> None:
    worker = get_worker(dataset, app_config)
    worker.pre_compute()
    assert len(worker.parquet_files) == 34
    response = worker.compute()
    assert len(response["parquet_files"]) == 34
    assert response["parquet_files"][0]["num_bytes"] == worker.parquet_files[0].size
    assert response["parquet_files"][0]["num_rows"] == 1104


# @pytest.mark.real_dataset
# @pytest.mark.parametrize("dataset", ["glue"])
# def test_process(app_config: AppConfig, dataset: str) -> None:
#     worker = get_worker(dataset, app_config)
#     assert worker.process() is True
#     cached_response = get_response(kind=worker.processing_step.cache_kind, dataset=dataset)
#     assert cached_response["http_status"] == HTTPStatus.OK
#     assert cached_response["error_code"] is None
#     assert cached_response["worker_version"] == worker.get_version()
#     assert cached_response["dataset_git_revision"] is not None
#     assert cached_response["error_code"] is None
#     content = cached_response["content"]
#     assert len(content["parquet_files"]) > 1
#     assert content["parquet_files"][0]["num_bytes"] > 0
#     assert content["parquet_files"][0]["num_examples"] > 0
