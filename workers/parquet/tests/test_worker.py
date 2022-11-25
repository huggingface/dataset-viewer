# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus

import pytest
from libcommon.queue import _clean_queue_database
from libcommon.simple_cache import DoesNotExist, _clean_cache_database, get_response

from parquet.config import WorkerConfig
from parquet.utils import CacheKind
from parquet.worker import ParquetWorker

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


def should_skip_job(worker: ParquetWorker, hub_public_csv: str) -> None:
    dataset = hub_public_csv
    assert worker.should_skip_job(dataset=dataset) is False
    # we add an entry to the cache
    worker.compute(dataset=dataset)
    assert worker.should_skip_job(dataset=dataset) is True


def test_compute(worker: ParquetWorker, hub_datasets: HubDatasets) -> None:
    dataset = hub_datasets["public"]["name"]
    assert worker.compute(dataset=dataset) is True
    cached_response = get_response(kind=CacheKind.PARQUET.value, dataset=dataset)
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None
    assert cached_response["worker_version"] == worker.version
    assert cached_response["dataset_git_revision"] is not None
    content = cached_response["content"]
    assert len(content["parquet_files"]) == 1
    assert content == hub_datasets["public"]["parquet_response"]


def test_doesnotexist(worker: ParquetWorker) -> None:
    dataset = "doesnotexist"
    assert worker.compute(dataset=dataset) is False
    with pytest.raises(DoesNotExist):
        get_response(kind=CacheKind.PARQUET.value, dataset=dataset)


def test_not_supported(worker: ParquetWorker, hub_not_supported_csv: str) -> None:
    assert worker.compute(dataset=hub_not_supported_csv) is False
    cached_response = get_response(kind=CacheKind.PARQUET.value, dataset=hub_not_supported_csv)
    assert cached_response["http_status"] == HTTPStatus.NOT_IMPLEMENTED
    assert cached_response["error_code"] == "DatasetNotSupportedError"


def test_process_job(worker: ParquetWorker, hub_public_csv: str) -> None:
    worker.queue.add_job(dataset=hub_public_csv)
    result = worker.process_next_job()
    assert result is True
