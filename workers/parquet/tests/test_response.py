# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest
from libcommon.exceptions import CustomError

from parquet.config import WorkerConfig
from parquet.response import compute_parquet_response, parse_repo_filename

from .fixtures.hub import HubDatasets


@pytest.mark.parametrize(
    "name,error_code,cause",
    [
        ("public", None, None),
        ("audio", None, None),
        # ("gated", None, None),
        # ("private", None, None),
        ("empty", "EmptyDatasetError", "EmptyDatasetError"),
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
        assert result["parquet_response"] == expected_parquet_response
        assert result["dataset_git_revision"] is not None
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
