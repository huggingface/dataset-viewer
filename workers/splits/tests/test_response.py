# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest
from libcommon.exceptions import CustomError

from splits.config import WorkerConfig
from splits.response import get_splits_response

from .fixtures.hub import HubDatasets


@pytest.mark.parametrize(
    "name,use_token,error_code,cause",
    [
        ("public", False, None, None),
        ("audio", False, None, None),
        ("gated", True, None, None),
        ("private", True, None, None),
        ("empty", False, "EmptyDatasetError", "EmptyDatasetError"),
        ("does_not_exist", False, "DatasetNotFoundError", None),
        ("gated", False, "DatasetNotFoundError", None),
        ("private", False, "DatasetNotFoundError", None),
    ],
)
def test_get_splits_response_simple_csv(
    hub_datasets: HubDatasets, name: str, use_token: bool, error_code: str, cause: str, worker_config: WorkerConfig
) -> None:
    dataset = hub_datasets[name]["name"]
    expected_splits_response = hub_datasets[name]["splits_response"]
    if error_code is None:
        splits_response = get_splits_response(
            dataset=dataset,
            hf_endpoint=worker_config.common.hf_endpoint,
            hf_token=worker_config.common.hf_token if use_token else None,
        )
        assert splits_response == expected_splits_response
        return

    with pytest.raises(CustomError) as exc_info:
        get_splits_response(
            dataset=dataset,
            hf_endpoint=worker_config.common.hf_endpoint,
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
