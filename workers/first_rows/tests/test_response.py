# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest
from datasets.packaged_modules import csv
from libutils.exceptions import CustomError

from first_rows.response import get_first_rows_response

from .fixtures.hub import HubDatasets
from .utils import ASSETS_BASE_URL, HF_ENDPOINT, HF_TOKEN, get_default_config_split


@pytest.mark.parametrize(
    "name,use_token,error_code,cause",
    [
        ("public", False, None, None),
        ("audio", False, None, None),
        ("image", False, None, None),
        ("images_list", False, None, None),
        ("jsonl", False, None, None),
        ("gated", True, None, None),
        ("private", True, None, None),
        ("empty", False, "EmptyDatasetError", "EmptyDatasetError"),
        ("does_not_exist", False, "DatasetNotFoundError", None),
        ("gated", False, "DatasetNotFoundError", None),
        ("private", False, "DatasetNotFoundError", None),
    ],
)
def test_number_rows(
    hub_datasets: HubDatasets,
    name: str,
    use_token: bool,
    error_code: str,
    cause: str,
) -> None:
    # temporary patch to remove the effect of
    # https://github.com/huggingface/datasets/issues/4875#issuecomment-1280744233
    # note: it fixes the tests, but it does not fix the bug in the "real world"
    if hasattr(csv, "_patched_for_streaming") and csv._patched_for_streaming:  # type: ignore
        csv._patched_for_streaming = False  # type: ignore

    dataset = hub_datasets[name]["name"]
    expected_first_rows_response = hub_datasets[name]["first_rows_response"]
    rows_max_number = 7
    dataset, config, split = get_default_config_split(dataset)
    if error_code is None:
        response = get_first_rows_response(
            dataset=dataset,
            config=config,
            split=split,
            assets_base_url=ASSETS_BASE_URL,
            hf_endpoint=HF_ENDPOINT,
            hf_token=HF_TOKEN if use_token else None,
            rows_max_number=rows_max_number,
        )
        assert response == expected_first_rows_response
        return
    with pytest.raises(CustomError) as exc_info:
        get_first_rows_response(
            dataset=dataset,
            config=config,
            split=split,
            assets_base_url=ASSETS_BASE_URL,
            hf_endpoint=HF_ENDPOINT,
            hf_token=HF_TOKEN if use_token else None,
            rows_max_number=rows_max_number,
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
