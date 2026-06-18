# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from collections.abc import Mapping
from typing import Literal

import pandas as pd
import pytest
from datasets import Dataset
from datasets.packaged_modules.json.json import AGENT_TRACES_FEATURES
from datasets.table import embed_table_storage

from libcommon.constants import MAX_NUM_ROWS_PER_PAGE
from libcommon.dtos import Row, RowsContent
from libcommon.exceptions import TooBigContentError
from libcommon.storage_client import StorageClient
from libcommon.viewer_utils.rows import GetRowsContent, create_first_rows_response

from ..constants import (
    CI_HUB_ENDPOINT,
    DATASETS_NAMES,
    DEFAULT_COLUMN_NAME,
    DEFAULT_COLUMNS_MAX_NUMBER,
    DEFAULT_CONFIG,
    DEFAULT_MIN_CELL_BYTES,
    DEFAULT_REVISION,
    DEFAULT_ROWS_MAX_BYTES,
    DEFAULT_ROWS_MAX_NUMBER,
    DEFAULT_ROWS_MIN_NUMBER,
    DEFAULT_SPLIT,
    SOME_BYTES,
)
from ..types import DatasetFixture
from ..utils import get_dataset_rows_content


@pytest.mark.parametrize("dataset_name", DATASETS_NAMES)
def test_create_first_rows_response(
    storage_client_with_url_preparator: StorageClient,
    datasets_fixtures: Mapping[str, DatasetFixture],
    dataset_name: str,
) -> None:
    dataset_fixture = datasets_fixtures[dataset_name]
    dataset = dataset_fixture.dataset
    dataset = dataset.with_format("arrow").map(embed_table_storage)

    response = create_first_rows_response(
        dataset=dataset_name,
        revision=DEFAULT_REVISION,
        config=DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        storage_client=storage_client_with_url_preparator,
        hf_endpoint=CI_HUB_ENDPOINT,
        hf_token=None,
        features=dataset.features,
        get_rows_content=get_dataset_rows_content(dataset=dataset),
        min_cell_bytes=DEFAULT_MIN_CELL_BYTES,
        rows_max_bytes=DEFAULT_ROWS_MAX_BYTES,
        rows_max_number=DEFAULT_ROWS_MAX_NUMBER,
        rows_min_number=DEFAULT_ROWS_MIN_NUMBER,
        columns_max_number=DEFAULT_COLUMNS_MAX_NUMBER,
    )
    assert not response["truncated"]
    assert response["features"][0]["type"] == dataset_fixture.expected_feature_type
    assert response["rows"][0]["row"] == {DEFAULT_COLUMN_NAME: dataset_fixture.expected_cell}
    assert response["rows"][0]["truncated_cells"] == []


NUM_ROWS = 15


def get_agent_trace_rows_content(rows: list[Row]) -> GetRowsContent:
    def _get_rows_content(rows_max_number: int) -> RowsContent:
        rows_to_return = rows[:rows_max_number]
        return RowsContent(
            rows=rows_to_return,
            all_fetched=len(rows) <= rows_max_number,
            truncated_columns=[],
        )

    return _get_rows_content


def get_agent_trace_row(trace: object) -> Row:
    return {
        "harness": "hermes",
        "session_id": "hermes-session",
        "prompt": "Run pwd and date.",
        "messages": [{"role": "assistant", "content": "a" * 5_000}],
        "tools": [],
        "metadata": {"trace_type": "hermes"},
        "sent_at": "2026-06-05T13:22:48.307Z",
        "num_user_messages": 1,
        "num_tool_calls": 1,
        "trace": trace,
        "file_path": "sessions.jsonl",
    }


def test_create_first_rows_response_keeps_agent_trace_when_it_fits(storage_client: StorageClient) -> None:
    trace = [{"role": "user", "content": "keep this row-level trace"}]

    response = create_first_rows_response(
        dataset="dataset",
        revision=DEFAULT_REVISION,
        config=DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        storage_client=storage_client,
        hf_endpoint=CI_HUB_ENDPOINT,
        hf_token=None,
        features=AGENT_TRACES_FEATURES,
        get_rows_content=get_agent_trace_rows_content([get_agent_trace_row(trace)]),
        min_cell_bytes=DEFAULT_MIN_CELL_BYTES,
        rows_max_bytes=2_000,
        rows_max_number=DEFAULT_ROWS_MAX_NUMBER,
        rows_min_number=DEFAULT_ROWS_MIN_NUMBER,
        columns_max_number=DEFAULT_COLUMNS_MAX_NUMBER,
    )

    row = response["rows"][0]
    assert row["row"]["trace"] == trace
    assert "trace" not in row["truncated_cells"]
    assert "messages" in row["truncated_cells"]


def test_create_first_rows_response_truncates_agent_trace_when_it_does_not_fit(
    storage_client: StorageClient,
) -> None:
    trace = [{"role": "user", "content": "x" * 5_000}]

    response = create_first_rows_response(
        dataset="dataset",
        revision=DEFAULT_REVISION,
        config=DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        storage_client=storage_client,
        hf_endpoint=CI_HUB_ENDPOINT,
        hf_token=None,
        features=AGENT_TRACES_FEATURES,
        get_rows_content=get_agent_trace_rows_content([get_agent_trace_row(trace)]),
        min_cell_bytes=DEFAULT_MIN_CELL_BYTES,
        rows_max_bytes=2_000,
        rows_max_number=DEFAULT_ROWS_MAX_NUMBER,
        rows_min_number=DEFAULT_ROWS_MIN_NUMBER,
        columns_max_number=DEFAULT_COLUMNS_MAX_NUMBER,
    )

    row = response["rows"][0]
    assert isinstance(row["row"]["trace"], str)
    assert "trace" in row["truncated_cells"]


@pytest.mark.parametrize(
    "rows_max_bytes,rows_max_number,truncated",
    [
        (1_000, NUM_ROWS + 5, True),  # truncated because of rows_max_bytes
        (10_000_000_000, NUM_ROWS - 5, True),  # truncated because of rows_max_number
        (10_000_000_000, NUM_ROWS + 5, False),  # not truncated
    ],
)
def test_create_first_rows_response_truncated(
    storage_client: StorageClient,
    rows_max_bytes: int,
    rows_max_number: int,
    truncated: bool,
) -> None:
    CELL_SIZE = 1_234
    dataset = Dataset.from_pandas(
        pd.DataFrame(
            ["a" * CELL_SIZE for _ in range(NUM_ROWS)],
            dtype=pd.StringDtype(storage="python"),
        )
    )

    response = create_first_rows_response(
        dataset="dataset",
        revision=DEFAULT_REVISION,
        config=DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        storage_client=storage_client,
        hf_endpoint=CI_HUB_ENDPOINT,
        hf_token=None,
        features=dataset.features,
        get_rows_content=get_dataset_rows_content(dataset=dataset),
        min_cell_bytes=DEFAULT_MIN_CELL_BYTES,
        rows_max_bytes=rows_max_bytes,
        rows_max_number=rows_max_number,
        rows_min_number=DEFAULT_ROWS_MIN_NUMBER,
        columns_max_number=DEFAULT_COLUMNS_MAX_NUMBER,
    )
    assert response["truncated"] == truncated


@pytest.mark.parametrize(
    "dataset_name,rows_max_bytes,expected",
    [
        # with rows_max_bytes > response size, the response is not truncated
        ("audio", 337 + SOME_BYTES, "complete"),
        ("image", 319 + SOME_BYTES, "complete"),
        ("urls", 268 + SOME_BYTES, "complete"),
        ("images_list", 455 + SOME_BYTES, "complete"),
        ("audios_list", 447 + SOME_BYTES, "complete"),
        ("images_sequence", 484 + SOME_BYTES, "complete"),
        ("audios_sequence", 476 + SOME_BYTES, "complete"),
        ("dict_of_audios_and_images", 940 + SOME_BYTES, "complete"),
        ("pdf", 8810 + SOME_BYTES, "complete"),
        # with rows_max_bytes < response size, the response is:
        # - not truncated for top-level Audio and Image features and urls
        # - truncated for nested Audio and Image features
        ("audio", 337 - SOME_BYTES, "complete"),
        ("image", 319 - SOME_BYTES, "complete"),
        ("urls", 268 - SOME_BYTES, "complete"),
        ("pdf", 8810 - SOME_BYTES, "complete"),
        ("images_list", 455 - SOME_BYTES, "truncated_cells"),
        ("audios_list", 447 - SOME_BYTES, "truncated_cells"),
        ("images_sequence", 484 - SOME_BYTES, "truncated_cells"),
        ("audios_sequence", 476 - SOME_BYTES, "truncated_cells"),
        ("dict_of_audios_and_images", 940 - SOME_BYTES, "truncated_cells"),
        # with rows_max_bytes <<< response size, a TooBigContentError exception is raised
        # (note that it should never happen if the correct set of parameters is chosen)
        ("audio", 10, "error"),
        ("image", 10, "error"),
        ("urls", 10, "error"),
        ("images_list", 10, "error"),
        ("audios_list", 10, "error"),
        ("images_sequence", 10, "error"),
        ("audios_sequence", 10, "error"),
        ("dict_of_audios_and_images", 10, "error"),
        ("pdf", 10, "error"),
    ],
)
def test_create_first_rows_response_truncation_on_audio_or_image(
    storage_client_with_url_preparator: StorageClient,
    datasets_fixtures: Mapping[str, DatasetFixture],
    dataset_name: str,
    rows_max_bytes: int,
    expected: Literal["error", "truncated_cells", "complete"],
) -> None:
    dataset_fixture = datasets_fixtures[dataset_name]
    dataset = dataset_fixture.dataset
    dataset = dataset.with_format("arrow").map(embed_table_storage)

    if expected == "error":
        with pytest.raises(TooBigContentError):
            response = create_first_rows_response(
                dataset=dataset_name,
                revision=DEFAULT_REVISION,
                config=DEFAULT_CONFIG,
                split=DEFAULT_SPLIT,
                storage_client=storage_client_with_url_preparator,
                hf_endpoint=CI_HUB_ENDPOINT,
                hf_token=None,
                features=dataset.features,
                get_rows_content=get_dataset_rows_content(dataset=dataset),
                min_cell_bytes=DEFAULT_MIN_CELL_BYTES,
                rows_max_bytes=rows_max_bytes,
                rows_max_number=MAX_NUM_ROWS_PER_PAGE,
                rows_min_number=DEFAULT_ROWS_MIN_NUMBER,
                columns_max_number=DEFAULT_COLUMNS_MAX_NUMBER,
            )
            print(response)
    else:
        response = create_first_rows_response(
            dataset=dataset_name,
            revision=DEFAULT_REVISION,
            config=DEFAULT_CONFIG,
            split=DEFAULT_SPLIT,
            storage_client=storage_client_with_url_preparator,
            hf_endpoint=CI_HUB_ENDPOINT,
            hf_token=None,
            features=dataset.features,
            get_rows_content=get_dataset_rows_content(dataset=dataset),
            min_cell_bytes=DEFAULT_MIN_CELL_BYTES,
            rows_max_bytes=rows_max_bytes,
            rows_max_number=MAX_NUM_ROWS_PER_PAGE,
            rows_min_number=DEFAULT_ROWS_MIN_NUMBER,
            columns_max_number=DEFAULT_COLUMNS_MAX_NUMBER,
        )
        assert not response["truncated"]
        # ^ no rows have been deleted
        assert len(response["rows"][0]["truncated_cells"]) == (1 if (expected == "truncated_cells") else 0), response
