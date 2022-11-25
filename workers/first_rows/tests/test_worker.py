# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.


from http import HTTPStatus

import pytest
from libcommon.processing_steps import first_rows_step
from libcommon.queue import _clean_queue_database
from libcommon.simple_cache import DoesNotExist, _clean_cache_database, get_response

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


def should_skip_job(worker: FirstRowsWorker, hub_public_csv: str) -> None:
    dataset, config, split = get_default_config_split(hub_public_csv)
    assert worker.should_skip_job(dataset=dataset, config=config, split=split) is False
    # we add an entry to the cache
    worker.compute(dataset=dataset, config=config, split=split)
    assert worker.should_skip_job(dataset=dataset, config=config, split=split) is True
    assert worker.should_skip_job(dataset=dataset, config=config, split=split, force=False) is False


def test_compute(worker: FirstRowsWorker, hub_public_csv: str) -> None:
    dataset, config, split = get_default_config_split(hub_public_csv)
    assert worker.compute(dataset=dataset, config=config, split=split) is True
    cached_response = get_response(kind=first_rows_step.cache_kind, dataset=dataset, config=config, split=split)
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None
    assert cached_response["worker_version"] == worker.version
    assert cached_response["dataset_git_revision"] is not None
    content = cached_response["content"]
    assert content["features"][0]["feature_idx"] == 0
    assert content["features"][0]["name"] == "col_1"
    assert content["features"][0]["type"]["_type"] == "Value"
    assert content["features"][0]["type"]["dtype"] == "int64"  # <---|
    assert content["features"][1]["type"]["dtype"] == "int64"  # <---|- auto-detected by the datasets library
    assert content["features"][2]["type"]["dtype"] == "float64"  # <-|


def test_doesnotexist(worker: FirstRowsWorker) -> None:
    dataset = "doesnotexist"
    dataset, config, split = get_default_config_split(dataset)
    assert worker.compute(dataset=dataset, config=config, split=split) is False
    with pytest.raises(DoesNotExist):
        get_response(kind=first_rows_step.cache_kind, dataset=dataset, config=config, split=split)


def test_process_job(worker: FirstRowsWorker, hub_public_csv: str) -> None:
    dataset, config, split = get_default_config_split(hub_public_csv)
    worker.queue.add_job(dataset=dataset, config=config, split=split)
    result = worker.process_next_job()
    assert result is True
