# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus

import pytest
from libcommon.queue import _clean_queue_database
from libcommon.simple_cache import DoesNotExist, _clean_cache_database, get_response

from splits.config import WorkerConfig
from splits.worker import SplitsWorker


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    _clean_cache_database()
    _clean_queue_database()


@pytest.fixture(autouse=True, scope="module")
def worker(worker_config: WorkerConfig) -> SplitsWorker:
    return SplitsWorker(worker_config)


def test_version(worker: SplitsWorker) -> None:
    assert len(worker.version.split(".")) == 3
    assert worker.compare_major_version(other_version="0.0.0") > 0
    assert worker.compare_major_version(other_version="1000.0.0") < 0


def should_skip_job(worker: SplitsWorker, hub_public_csv: str) -> None:
    dataset = hub_public_csv
    assert worker.should_skip_job(dataset=dataset) is False
    # we add an entry to the cache
    worker.compute(dataset=dataset)
    assert worker.should_skip_job(dataset=dataset) is True
    assert worker.should_skip_job(dataset=dataset, force=True) is False


def test_compute(worker: SplitsWorker, hub_public_csv: str) -> None:
    dataset = hub_public_csv
    assert worker.compute(dataset=dataset) is True
    cached_response = get_response(kind=worker.processing_step.cache_kind, dataset=hub_public_csv)
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None
    assert cached_response["worker_version"] == worker.version
    assert cached_response["dataset_git_revision"] is not None
    assert cached_response["error_code"] is None
    content = cached_response["content"]
    assert len(content["splits"]) == 1
    assert content["splits"][0]["num_bytes"] is None
    assert content["splits"][0]["num_examples"] is None


def test_doesnotexist(worker: SplitsWorker) -> None:
    dataset = "doesnotexist"
    assert worker.compute(dataset=dataset) is False
    with pytest.raises(DoesNotExist):
        get_response(kind=worker.processing_step.cache_kind, dataset=dataset)


def test_process_job(worker: SplitsWorker, hub_public_csv: str) -> None:
    worker.queue.add_job(dataset=hub_public_csv)
    result = worker.process_next_job()
    assert result is True
