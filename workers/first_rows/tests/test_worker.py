# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.


from http import HTTPStatus

import pytest
from libcache.simple_cache import DoesNotExist
from libcache.simple_cache import _clean_database as _clean_cache_database
from libcache.simple_cache import get_first_rows_response
from libqueue.queue import _clean_queue_database

from first_rows.config import WorkerConfig
from first_rows.worker import FirstRowsWorker

from .utils import get_default_config_split


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    _clean_cache_database()
    _clean_queue_database()


@pytest.fixture(autouse=True, scope="module")
def worker(worker_config: WorkerConfig) -> FirstRowsWorker:
    return FirstRowsWorker(worker_config)


def test_compute(worker: FirstRowsWorker, hub_public_csv: str) -> None:
    dataset, config, split = get_default_config_split(hub_public_csv)
    assert worker.compute(dataset=dataset, config=config, split=split) is True
    cache_entry = get_first_rows_response(dataset_name=dataset, config_name=config, split_name=split)
    assert cache_entry["http_status"] == HTTPStatus.OK
    assert cache_entry["error_code"] is None
    assert cache_entry["worker_version"] == worker.version
    assert cache_entry["dataset_git_revision"] is not None
    response = cache_entry["response"]
    assert response["features"][0]["feature_idx"] == 0
    assert response["features"][0]["name"] == "col_1"
    assert response["features"][0]["type"]["_type"] == "Value"
    assert response["features"][0]["type"]["dtype"] == "int64"  # <---|
    assert response["features"][1]["type"]["dtype"] == "int64"  # <---|- auto-detected by the datasets library
    assert response["features"][2]["type"]["dtype"] == "float64"  # <-|


def test_doesnotexist(worker: FirstRowsWorker) -> None:
    dataset = "doesnotexist"
    dataset, config, split = get_default_config_split(dataset)
    assert worker.compute(dataset=dataset, config=config, split=split) is False
    with pytest.raises(DoesNotExist):
        get_first_rows_response(dataset_name=dataset, config_name=config, split_name=split)


def test_process_job(worker: FirstRowsWorker, hub_public_csv: str) -> None:
    dataset, config, split = get_default_config_split(hub_public_csv)
    worker.queue.add_job(dataset=dataset, config=config, split=split)
    result = worker.process_next_job()
    assert result is True
