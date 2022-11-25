# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus

import pytest
from libcommon.exceptions import CustomError
from libcommon.queue import _clean_queue_database
from libcommon.simple_cache import DoesNotExist, _clean_cache_database, get_response

from splits.config import WorkerConfig
from splits.worker import SplitsWorker, compute_splits_response

from .fixtures.hub import HubDatasets


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
    worker.process(dataset=dataset)
    assert worker.should_skip_job(dataset=dataset) is True
    assert worker.should_skip_job(dataset=dataset, force=True) is False


def test_process(worker: SplitsWorker, hub_public_csv: str) -> None:
    dataset = hub_public_csv
    assert worker.process(dataset=dataset) is True
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
    assert worker.process(dataset=dataset) is False
    with pytest.raises(DoesNotExist):
        get_response(kind=worker.processing_step.cache_kind, dataset=dataset)


def test_process_job(worker: SplitsWorker, hub_public_csv: str) -> None:
    worker.queue.add_job(dataset=hub_public_csv)
    result = worker.process_next_job()
    assert result is True


@pytest.mark.parametrize(
    "name,use_token,error_code,cause",
    [
        ("public", False, None, None),
        ("audio", False, None, None),
        ("gated", True, None, None),
        ("private", True, None, None),
        ("empty", False, "EmptyDatasetError", "EmptyDatasetError"),
        # should we really test the following cases?
        # The assumption is that the dataset exists and is accessible with the token
        ("does_not_exist", False, "SplitsNamesError", "FileNotFoundError"),
        ("gated", False, "SplitsNamesError", "FileNotFoundError"),
        ("private", False, "SplitsNamesError", "FileNotFoundError"),
    ],
)
def test_compute_splits_response_simple_csv(
    hub_datasets: HubDatasets, name: str, use_token: bool, error_code: str, cause: str, worker_config: WorkerConfig
) -> None:
    dataset = hub_datasets[name]["name"]
    expected_splits_response = hub_datasets[name]["splits_response"]
    if error_code is None:
        result = compute_splits_response(
            dataset=dataset,
            hf_token=worker_config.common.hf_token if use_token else None,
        )
        assert result == expected_splits_response
        return

    with pytest.raises(CustomError) as exc_info:
        compute_splits_response(
            dataset=dataset,
            hf_token=worker_config.common.hf_token if use_token else None,
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
