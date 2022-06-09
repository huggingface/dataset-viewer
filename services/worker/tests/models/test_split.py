import json
import pandas
import pytest

from libutils.types import RowItem, RowsResponse
from worker.models.split import get_split, get_json_split, to_json_rows_response

from .._utils import HF_TOKEN, ROWS_MAX_NUMBER

# TODO: test fallback


def test_get_split() -> None:
    dataset_name = "acronym_identification"
    config_name = "default"
    split_name = "train"
    split = get_split(dataset_name, config_name, split_name)

    assert split["num_bytes"] == 7792803
    assert split["num_examples"] == 14006


def test_gated() -> None:
    dataset_name = "severo/dummy_gated"
    config_name = "severo--embellishments"
    split_name = "train"
    split = get_split(dataset_name, config_name, split_name, HF_TOKEN, rows_max_number=ROWS_MAX_NUMBER)
    assert len(split["rows_response"]["rows"]) == ROWS_MAX_NUMBER
    assert split["rows_response"]["rows"][0]["row"]["year"] == "1855"

    json_split = get_json_split(dataset_name, config_name, split_name, HF_TOKEN, rows_max_number=ROWS_MAX_NUMBER)
    output_rows_response = json.loads(json_split["json_rows_response"])
    assert len(output_rows_response["rows"]) == ROWS_MAX_NUMBER
    assert output_rows_response["rows"][0]["row"]["year"] == "1855"


def test_fallback() -> None:
    # https://github.com/huggingface/datasets/issues/3185
    dataset_name = "samsum"
    config_name = "samsum"
    split_name = "train"
    MAX_SIZE_FALLBACK = 100_000_000
    split = get_split(
        dataset_name,
        config_name,
        split_name,
        HF_TOKEN,
        rows_max_number=ROWS_MAX_NUMBER,
        max_size_fallback=MAX_SIZE_FALLBACK,
    )
    assert len(split["rows_response"]["rows"]) == ROWS_MAX_NUMBER


def test_pandas_timestamp() -> None:
    # see https://github.com/huggingface/datasets/issues/4413
    timestamp = pandas.Timestamp(2022, 5, 1)
    row: RowItem = {
        "dataset": "test",
        "config": "test",
        "split": "test",
        "row_idx": 0,
        "row": {"col1": timestamp},
        "truncated_cells": [],
    }
    rows_response: RowsResponse = {"rows": [row], "columns": []}
    # not supported
    with pytest.raises(TypeError):
        to_json_rows_response(rows_response)


# TODO: test the truncation
