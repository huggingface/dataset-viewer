# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.


import itertools
from collections.abc import Mapping
from typing import Any

import pytest

from libcommon.dtos import RowItem, RowsContent
from libcommon.storage_client import StorageClient
from libcommon.viewer_utils.rows import create_first_rows_response, truncate_row_item

from ..constants import (
    DATASETS_NAMES,
    DEFAULT_COLUMN_NAME,
    DEFAULT_CONFIG,
    DEFAULT_REVISION,
    DEFAULT_SPLIT,
    TEN_CHARS_TEXT,
)
from ..types import DatasetFixture


@pytest.mark.parametrize(
    "cells, min_cell_bytes, columns_to_keep_untruncated, expected_cells, expected_truncated_cells",
    [
        ([TEN_CHARS_TEXT], 11, [], ['"' + TEN_CHARS_TEXT], ["c0"]),
        ([TEN_CHARS_TEXT], 11, ["c0"], [TEN_CHARS_TEXT], []),
        ([TEN_CHARS_TEXT], 12, [], [TEN_CHARS_TEXT], []),
        ([TEN_CHARS_TEXT, TEN_CHARS_TEXT], 11, [], ['"' + TEN_CHARS_TEXT, '"' + TEN_CHARS_TEXT], ["c0", "c1"]),
        ([TEN_CHARS_TEXT, TEN_CHARS_TEXT], 11, ["c1"], ['"' + TEN_CHARS_TEXT, TEN_CHARS_TEXT], ["c0"]),
        ([{"a": 1}], 11, [], [{"a": 1}], []),
        ([{"a": "b" * 100}], 11, [], ['{"a":"bbbbb'], ["c0"]),
    ],
)
def test_truncate_row_item(
    cells: list[Any],
    min_cell_bytes: int,
    columns_to_keep_untruncated: list[str],
    expected_cells: list[Any],
    expected_truncated_cells: list[str],
) -> None:
    row_item = RowItem(row_idx=0, row={f"c{i}": cell for i, cell in enumerate(cells)}, truncated_cells=[])
    truncated_row_item = truncate_row_item(row_item, min_cell_bytes, columns_to_keep_untruncated)
    assert truncated_row_item["truncated_cells"] == expected_truncated_cells
    for i, cell in enumerate(expected_cells):
        assert truncated_row_item["row"][f"c{i}"] == cell


@pytest.mark.parametrize("dataset_name", DATASETS_NAMES)
def test_create_first_rows_response(
    storage_client: StorageClient, datasets_fixtures: Mapping[str, DatasetFixture], dataset_name: str
) -> None:
    dataset_fixture = datasets_fixtures[dataset_name]
    dataset = dataset_fixture.dataset

    def get_rows_content(rows_max_number: int) -> RowsContent:
        rows_plus_one = list(itertools.islice(dataset, rows_max_number + 1))
        # ^^ to be able to detect if a split has exactly ROWS_MAX_NUMBER rows
        return RowsContent(rows=rows_plus_one[:rows_max_number], all_fetched=len(rows_plus_one) <= rows_max_number)

    response = create_first_rows_response(
        dataset=dataset_name,
        revision=DEFAULT_REVISION,
        config=DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        storage_client=storage_client,
        features=dataset.features,
        get_rows_content=get_rows_content,
        min_cell_bytes=0,
        rows_max_bytes=1000000,
        rows_max_number=1000000,
        rows_min_number=0,
        columns_max_number=100000,
    )
    assert response["features"][0]["type"] == dataset_fixture.expected_feature_type
    assert response["rows"][0]["row"] == {DEFAULT_COLUMN_NAME: dataset_fixture.expected_cell}
