# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus

import pytest
from libcache.simple_cache import DoesNotExist
from libcache.simple_cache import _clean_database as _clean_cache_database
from libcache.simple_cache import get_splits_response
from libqueue.queue import _clean_queue_database

from splits.config import WorkerConfig
from splits.worker import SplitsWorker


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    _clean_cache_database()
    _clean_queue_database()


@pytest.fixture(autouse=True, scope="module")
def worker(worker_config: WorkerConfig) -> SplitsWorker:
    return SplitsWorker(worker_config)


def test_compute(worker: SplitsWorker, hub_public_csv: str) -> None:
    dataset = hub_public_csv
    assert worker.compute(dataset=dataset) is True
    response, cached_http_status, error_code = get_splits_response(dataset_name=hub_public_csv)
    assert cached_http_status == HTTPStatus.OK
    assert error_code is None
    assert len(response["splits"]) == 1
    assert response["splits"][0]["num_bytes"] is None
    assert response["splits"][0]["num_examples"] is None


def test_doesnotexist(worker: SplitsWorker) -> None:
    dataset = "doesnotexist"
    assert worker.compute(dataset=dataset) is False
    with pytest.raises(DoesNotExist):
        get_splits_response(dataset_name=dataset)


def test_process_job(worker: SplitsWorker, hub_public_csv: str) -> None:
    worker.queue.add_job(dataset=hub_public_csv)
    result = worker.process_next_job()
    assert result is True
