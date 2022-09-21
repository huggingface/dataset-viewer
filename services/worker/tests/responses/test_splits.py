# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest
from libutils.exceptions import CustomError

from worker.responses.splits import get_splits_response

from ..fixtures.hub import HubDatasets
from ..utils import HF_ENDPOINT, HF_TOKEN


@pytest.mark.parametrize(
    "name,use_token,error_code,cause",
    [
        ("public", False, None, None),
        ("audio", False, None, None),
        ("gated", True, None, None),
        ("private", True, None, None),
        ("empty", False, "SplitsNamesError", "EmptyDatasetError"),
        ("does_not_exist", False, "DatasetNotFoundError", None),
        ("gated", False, "SplitsNamesError", "FileNotFoundError"),
        ("private", False, "SplitsNamesError", "FileNotFoundError"),
    ],
)
def test_get_splits_response_simple_csv(
    hub_datasets: HubDatasets, name: str, use_token: bool, error_code: str, cause: str
) -> None:
    dataset = hub_datasets[name]["name"]
    expected_splits_response = hub_datasets[name]["splits_response"]
    if error_code is None:
        splits_response = get_splits_response(dataset, HF_ENDPOINT, HF_TOKEN if use_token else None)
        assert splits_response == expected_splits_response
        return

    with pytest.raises(CustomError) as exc_info:
        get_splits_response(dataset, HF_ENDPOINT, HF_TOKEN if use_token else None)
    assert exc_info.value.code == error_code
    if cause is None:
        assert exc_info.value.disclose_cause is False
        assert exc_info.value.cause_exception is None
    else:
        assert exc_info.value.disclose_cause is True
        assert exc_info.value.cause_exception == cause
        response = exc_info.value.as_response()
        assert set(response.keys()) == {"error", "cause_exception", "cause_message", "cause_traceback"}
        assert response["error"] == "Cannot get the split names for the dataset."
        response_dict = dict(response)
        # ^ to remove mypy warnings
        assert response_dict["cause_exception"] == cause
        assert isinstance(response_dict["cause_traceback"], list)
        assert response_dict["cause_traceback"][0] == "Traceback (most recent call last):\n"
