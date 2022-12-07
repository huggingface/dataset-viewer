# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import io
from http import HTTPStatus
from typing import Iterator

import pandas as pd
import pytest
import requests
from libcommon.exceptions import CustomError
from libcommon.simple_cache import DoesNotExist, get_response

from datasets_based.config import AppConfig
from datasets_based.workers.parquet import (
    ParquetWorker,
    compute_parquet_response,
    parse_repo_filename,
)

from ..fixtures.hub import HubDatasets


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@pytest.fixture(scope="module", autouse=True)
def set_supported_datasets(hub_datasets: HubDatasets) -> Iterator[pytest.MonkeyPatch]:
    mp = pytest.MonkeyPatch()
    mp.setenv(
        "PARQUET_SUPPORTED_DATASETS",
        ",".join(value["name"] for value in hub_datasets.values() if "big" not in value["name"]),
    )
    yield mp
    mp.undo()


@pytest.fixture
def worker(app_config: AppConfig) -> ParquetWorker:
    return ParquetWorker(app_config=app_config)


def test_compute(worker: ParquetWorker, hub_datasets: HubDatasets) -> None:
    dataset = hub_datasets["public"]["name"]
    assert worker.process(dataset=dataset) is True
    cached_response = get_response(kind=worker.processing_step.cache_kind, dataset=dataset)
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
        get_response(kind=worker.processing_step.cache_kind, dataset=dataset)


def test_not_supported(worker: ParquetWorker, hub_public_big: str) -> None:
    # Not in the list of supported datasets and bigger than the maximum size
    assert worker.process(dataset=hub_public_big) is False
    cached_response = get_response(kind=worker.processing_step.cache_kind, dataset=hub_public_big)
    assert cached_response["http_status"] == HTTPStatus.NOT_IMPLEMENTED
    assert cached_response["error_code"] == "DatasetNotSupportedError"


def test_process_job(worker: ParquetWorker, hub_public_csv: str) -> None:
    worker.queue.add_job(dataset=hub_public_csv)
    result = worker.process_next_job()
    assert result is True


@pytest.mark.parametrize(
    "name",
    [
        "public",
        "audio",
        "gated",
        # working because the user has created these datasets, thus has access
        # TODO: test on a dataset that does not belong to the user
        # ("gated_extra_fields", "GatedExtraFieldsError", None),
        # ("private", "DatasetNotFoundError", None),
        "gated_extra_fields",
        "private",
    ],
)
def test_compute_splits_response_simple_csv_ok(
    hub_datasets: HubDatasets, name: str, app_config: AppConfig, data_df: pd.DataFrame
) -> None:
    dataset = hub_datasets[name]["name"]
    expected_parquet_response = hub_datasets[name]["parquet_response"]
    result = compute_parquet_response(
        dataset=dataset,
        hf_endpoint=app_config.common.hf_endpoint,
        hf_token=app_config.common.hf_token,
        committer_hf_token=app_config.parquet.committer_hf_token,
        source_revision=app_config.parquet.source_revision,
        target_revision=app_config.parquet.target_revision,
        commit_message=app_config.parquet.commit_message,
        url_template=app_config.parquet.url_template,
        supported_datasets=app_config.parquet.supported_datasets,
        max_dataset_size=app_config.parquet.max_dataset_size,
    )
    assert result == expected_parquet_response

    # download the parquet file and check that it is valid
    if name == "audio":
        return

    if name == "public":
        df = pd.read_parquet(result["parquet_files"][0]["url"], engine="auto")
    else:
        # in all these cases, the parquet files are not accessible without a token
        with pytest.raises(Exception):
            pd.read_parquet(result["parquet_files"][0]["url"], engine="auto")
        r = requests.get(
            result["parquet_files"][0]["url"], headers={"Authorization": f"Bearer {app_config.common.hf_token}"}
        )
        assert r.status_code == HTTPStatus.OK, r.text
        df = pd.read_parquet(io.BytesIO(r.content), engine="auto")
    assert df.equals(data_df), df


@pytest.mark.parametrize(
    "name,error_code,cause",
    [
        ("empty", "EmptyDatasetError", "EmptyDatasetError"),
        ("does_not_exist", "DatasetNotFoundError", None),
    ],
)
def test_compute_splits_response_simple_csv_error(
    hub_datasets: HubDatasets, name: str, error_code: str, cause: str, app_config: AppConfig
) -> None:
    dataset = hub_datasets[name]["name"]
    with pytest.raises(CustomError) as exc_info:
        compute_parquet_response(
            dataset=dataset,
            hf_endpoint=app_config.common.hf_endpoint,
            hf_token=app_config.common.hf_token,
            committer_hf_token=app_config.parquet.committer_hf_token,
            source_revision=app_config.parquet.source_revision,
            target_revision=app_config.parquet.target_revision,
            commit_message=app_config.parquet.commit_message,
            url_template=app_config.parquet.url_template,
            supported_datasets=app_config.parquet.supported_datasets,
            max_dataset_size=app_config.parquet.max_dataset_size,
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
