# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus

import pytest
from libcommon.exceptions import CustomError
from libcommon.processing_steps import parquet_step
from libcommon.queue import _clean_queue_database
from libcommon.simple_cache import DoesNotExist, _clean_cache_database, get_response

from parquet.config import WorkerConfig
from parquet.worker import ParquetWorker, compute_parquet_response, parse_repo_filename

from .fixtures.hub import HubDatasets


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    _clean_cache_database()
    _clean_queue_database()


@pytest.fixture(autouse=True, scope="module")
def worker(worker_config: WorkerConfig) -> ParquetWorker:
    return ParquetWorker(worker_config)


def test_version(worker: ParquetWorker) -> None:
    assert len(worker.version.split(".")) == 3
    assert worker.compare_major_version(other_version="0.0.0") > 0
    assert worker.compare_major_version(other_version="1000.0.0") < 0


def test_compute(worker: ParquetWorker, hub_datasets: HubDatasets) -> None:
    dataset = hub_datasets["public"]["name"]
    assert worker.process(dataset=dataset) is True
    cached_response = get_response(kind=parquet_step.cache_kind, dataset=dataset)
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None
    assert cached_response["worker_version"] == worker.version
    assert cached_response["dataset_git_revision"] is not None
    content = cached_response["content"]
    assert len(content["parquet_files"]) == 1
    assert content == hub_datasets["public"]["parquet_response"]


def test_doesnotexist(worker: ParquetWorker) -> None:
    dataset = "doesnotexist"
    assert worker.process(dataset=dataset) is False
    with pytest.raises(DoesNotExist):
        get_response(kind=parquet_step.cache_kind, dataset=dataset)


def test_not_supported(worker: ParquetWorker, hub_not_supported_csv: str) -> None:
    assert worker.process(dataset=hub_not_supported_csv) is False
    cached_response = get_response(kind=parquet_step.cache_kind, dataset=hub_not_supported_csv)
    assert cached_response["http_status"] == HTTPStatus.NOT_IMPLEMENTED
    assert cached_response["error_code"] == "DatasetNotSupportedError"


def test_process_job(worker: ParquetWorker, hub_public_csv: str) -> None:
    worker.queue.add_job(dataset=hub_public_csv)
    result = worker.process_next_job()
    assert result is True


@pytest.mark.parametrize(
    "name,error_code,cause",
    [
        ("public", None, None),
        ("audio", None, None),
        ("gated", None, None),
        ("gated_extra_fields", "GatedExtraFieldsError", None),
        ("empty", "EmptyDatasetError", "EmptyDatasetError"),
        ("private", "DatasetNotFoundError", None),
        ("does_not_exist", "DatasetNotFoundError", None),
    ],
)
def test_compute_splits_response_simple_csv(
    hub_datasets: HubDatasets, name: str, error_code: str, cause: str, worker_config: WorkerConfig
) -> None:
    dataset = hub_datasets[name]["name"]
    expected_parquet_response = hub_datasets[name]["parquet_response"]
    if error_code is None:
        result = compute_parquet_response(
            dataset=dataset,
            hf_endpoint=worker_config.common.hf_endpoint,
            hf_token=worker_config.parquet.hf_token,
            source_revision=worker_config.parquet.source_revision,
            target_revision=worker_config.parquet.target_revision,
            commit_message=worker_config.parquet.commit_message,
            url_template=worker_config.parquet.url_template,
            supported_datasets=worker_config.parquet.supported_datasets,
        )
        assert result == expected_parquet_response
        return

    with pytest.raises(CustomError) as exc_info:
        compute_parquet_response(
            dataset=dataset,
            hf_endpoint=worker_config.common.hf_endpoint,
            hf_token=worker_config.parquet.hf_token,
            source_revision=worker_config.parquet.source_revision,
            target_revision=worker_config.parquet.target_revision,
            commit_message=worker_config.parquet.commit_message,
            url_template=worker_config.parquet.url_template,
            supported_datasets=worker_config.parquet.supported_datasets,
        )
    assert exc_info.value.code == error_code
    if cause is None:
        assert exc_info.value.disclose_cause is False
        assert exc_info.value.cause_exception is None
    else:
        assert exc_info.value.disclose_cause is True
        assert exc_info.value.cause_exception == cause
        response = exc_info.value.as_response()
        assert set(response.keys()) == {"error", "cause_exception", "cause_message", "cause_traceback"}
        response_dict = dict(response)
        # ^ to remove mypy warnings
        assert response_dict["cause_exception"] == cause
        assert isinstance(response_dict["cause_traceback"], list)
        assert response_dict["cause_traceback"][0] == "Traceback (most recent call last):\n"


@pytest.mark.parametrize(
    "filename,split,config,raises",
    [
        ("config/builder-split.parquet", "split", "config", False),
        ("config/builder-split-00000-of-00001.parquet", "split", "config", False),
        ("builder-split-00000-of-00001.parquet", "split", "config", True),
        ("config/builder-not-supported.parquet", "not-supported", "config", True),
    ],
)
def test_parse_repo_filename(filename: str, split: str, config: str, raises: bool) -> None:
    if raises:
        with pytest.raises(Exception):
            parse_repo_filename(filename)
    else:
        assert parse_repo_filename(filename) == (config, split)
