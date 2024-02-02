# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.


from typing import Any

import pytest

from libcommon.dtos import RowItem
from libcommon.utils import get_json_size
from libcommon.viewer_utils.truncate_rows import (
    create_truncated_row_items,
    truncate_row_item,
    truncate_row_items_cells,
)

from ..constants import (
    DEFAULT_CELL_BYTES,
    DEFAULT_MIN_CELL_BYTES,
    DEFAULT_NUM_CELLS,
    DEFAULT_NUM_ROWS,
    DEFAULT_ROWS_MAX_BYTES,
    DEFAULT_ROWS_MIN_NUMBER,
    TEN_CHARS_TEXT,
)


@pytest.mark.parametrize(
    "cells,min_cell_bytes,columns_to_keep_untruncated,expected_cells,expected_truncated_cells",
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


def assert_test_truncate_row_items_cells(
    expected_num_truncated_rows: int,
    expected_is_under_limit: bool,
    num_rows: int = DEFAULT_NUM_ROWS,
    num_cells: int = DEFAULT_NUM_CELLS,
    cell_bytes: int = DEFAULT_CELL_BYTES,
    min_cell_bytes: int = DEFAULT_MIN_CELL_BYTES,
    rows_max_bytes: int = DEFAULT_ROWS_MAX_BYTES,
    columns_to_keep_untruncated: list[str] = [],
) -> None:
    row_items = [
        RowItem(row_idx=row_idx, row={f"c{i}": "a" * cell_bytes for i in range(num_cells)}, truncated_cells=[])
        # ^ the cell size is DEFAULT_CELL_BYTES, while the rest of the object accounts for other ~50 bytes (row_idx, truncated_cells)
        for row_idx in range(num_rows)
    ]
    truncated_row_items = truncate_row_items_cells(
        row_items=row_items,
        min_cell_bytes=min_cell_bytes,
        rows_max_bytes=rows_max_bytes,
        columns_to_keep_untruncated=columns_to_keep_untruncated,
    )
    assert (
        len([row_item for row_item in truncated_row_items if row_item["truncated_cells"]])
        == expected_num_truncated_rows
    )
    assert not any(
        row_item["truncated_cells"]
        for row_item in truncated_row_items[: len(truncated_row_items) - expected_num_truncated_rows]
    )
    # ^ the first rows are not truncated
    assert all(
        row_item["truncated_cells"]
        for row_item in truncated_row_items[len(truncated_row_items) - expected_num_truncated_rows :]
    )
    # ^ the last rows are truncated
    assert (get_json_size(truncated_row_items) <= rows_max_bytes) is expected_is_under_limit


@pytest.mark.parametrize(
    "num_rows,expected_num_truncated_rows,expected_is_under_limit",
    [
        # no truncated rows
        (1, 0, True),
        (10, 0, True),
        (20, 0, True),
        # some truncated rows
        (50, 28, True),
        (75, 73, True),
        (77, 76, True),
        # all truncated rows, but still under limit
        (78, 78, True),
        # all truncated rows, and over limit
        (79, 79, False),
        (100, 100, False),
    ],
)
# all other things being equal, increasing the number of rows increases the proportion of truncated rows
def test_truncate_row_items_cells_num_rows(
    num_rows: int,
    expected_num_truncated_rows: int,
    expected_is_under_limit: bool,
) -> None:
    assert_test_truncate_row_items_cells(
        expected_num_truncated_rows=expected_num_truncated_rows,
        expected_is_under_limit=expected_is_under_limit,
        num_rows=num_rows,
    )


@pytest.mark.parametrize(
    "num_cells,expected_num_truncated_rows,expected_is_under_limit",
    [
        # no truncated rows
        (1, 0, True),
        (2, 0, True),
        (3, 0, True),
        # some truncated rows
        (4, 3, True),
        (5, 6, True),
        (6, 8, True),
        (7, 9, True),
        (8, 10, True),
        (9, 11, True),
        (10, 11, True),
        (11, 12, True),
        (12, 12, True),
        # all truncated rows, but still under the limit
        (14, DEFAULT_NUM_ROWS, True),
        (15, DEFAULT_NUM_ROWS, True),
        # all truncated rows, and over limit
        (16, DEFAULT_NUM_ROWS, False),
        (17, DEFAULT_NUM_ROWS, False),
    ],
)
# all other things being equal, increasing the number of cells increases the proportion of truncated rows
def test_truncate_row_items_cells_num_cells(
    num_cells: int,
    expected_num_truncated_rows: int,
    expected_is_under_limit: bool,
) -> None:
    assert_test_truncate_row_items_cells(
        expected_num_truncated_rows=expected_num_truncated_rows,
        expected_is_under_limit=expected_is_under_limit,
        num_cells=num_cells,
    )


@pytest.mark.parametrize(
    "cell_bytes,expected_num_truncated_rows,expected_is_under_limit",
    [
        # no truncated rows
        (1, 0, True),
        (10, 0, True),
        (100, 0, True),
        (200, 0, True),
        # some truncated rows
        (500, 5, True),
        (1_000, 9, True),
        (2_000, 11, True),
        # all truncated rows, but still under the limit
        (5_000, DEFAULT_NUM_ROWS, True),
        (10_000, DEFAULT_NUM_ROWS, True),
        (10_000, DEFAULT_NUM_ROWS, True),
        # all truncated rows, and over limit
        # -> not possible
    ],
)
# all other things being equal, increasing the cell size increases the proportion of truncated rows
def test_truncate_row_items_cells_cell_bytes(
    cell_bytes: int,
    expected_num_truncated_rows: int,
    expected_is_under_limit: bool,
) -> None:
    assert_test_truncate_row_items_cells(
        expected_num_truncated_rows=expected_num_truncated_rows,
        expected_is_under_limit=expected_is_under_limit,
        cell_bytes=cell_bytes,
    )


@pytest.mark.parametrize(
    "min_cell_bytes,expected_num_truncated_rows,expected_is_under_limit",
    [
        # no truncated rows
        # -> not possible with the other settings
        # some truncated rows
        (1, 9, True),
        (10, 9, True),
        (100, 10, True),
        (200, 11, True),
        # all truncated rows, but still under the limit
        (300, DEFAULT_NUM_ROWS, True),
        # all truncated rows, and over limit
        (350, DEFAULT_NUM_ROWS, False),
        (500, DEFAULT_NUM_ROWS, False),
        # min_cell_bytes is higher than the cells, so no truncation happens, size is over the limit
        (1_000, 0, False),
        (2_000, 0, False),
    ],
)
# all other things being equal, increasing the minimum size of each cell increases the proportion of truncated rows
def test_truncate_row_items_cells_min_cell_bytes(
    min_cell_bytes: int,
    expected_num_truncated_rows: int,
    expected_is_under_limit: bool,
) -> None:
    assert_test_truncate_row_items_cells(
        expected_num_truncated_rows=expected_num_truncated_rows,
        expected_is_under_limit=expected_is_under_limit,
        min_cell_bytes=min_cell_bytes,
        cell_bytes=900,
    )


@pytest.mark.parametrize(
    "rows_max_bytes,expected_num_truncated_rows,expected_is_under_limit",
    [
        # no truncated rows
        (5_000, 0, True),
        # some truncated rows
        (1_000, 11, True),
        # all truncated rows, but still under the limit
        (850, DEFAULT_NUM_ROWS, True),
        # all truncated rows, and over limit
        (500, DEFAULT_NUM_ROWS, False),
        (100, DEFAULT_NUM_ROWS, False),
        (10, DEFAULT_NUM_ROWS, False),
        (1, DEFAULT_NUM_ROWS, False),
    ],
)
# all other things being equal, decreasing the maximum size of the rows increases the proportion of truncated rows
def test_truncate_row_items_cells_rows_max_bytes(
    rows_max_bytes: int,
    expected_num_truncated_rows: int,
    expected_is_under_limit: bool,
) -> None:
    assert_test_truncate_row_items_cells(
        expected_num_truncated_rows=expected_num_truncated_rows,
        expected_is_under_limit=expected_is_under_limit,
        rows_max_bytes=rows_max_bytes,
    )


@pytest.mark.parametrize(
    "num_columns_to_keep_untruncated,expected_num_truncated_rows,expected_is_under_limit",
    [
        # no truncated rows
        # <- not possible with the other settings
        # some truncated rows
        (0, 6, True),
        (1, 7, True),
        (2, 9, True),
        # all truncated rows, but still under the limit
        (3, DEFAULT_NUM_ROWS, True),
        # all truncated rows, and over limit
        (4, DEFAULT_NUM_ROWS, False),
        # all the columns are in the exception list, so no truncation happens, size is over the limit
        (5, 0, False),
    ],
)
# all other things being equal, increasing the number of columns to keep untruncated increases the proportion of truncated rows
def test_truncate_row_items_cells_untruncated_columns(
    num_columns_to_keep_untruncated: int,
    expected_num_truncated_rows: int,
    expected_is_under_limit: bool,
) -> None:
    assert_test_truncate_row_items_cells(
        expected_num_truncated_rows=expected_num_truncated_rows,
        expected_is_under_limit=expected_is_under_limit,
        columns_to_keep_untruncated=[f"c{i}" for i in range(num_columns_to_keep_untruncated)],
        num_cells=5,
    )


@pytest.mark.parametrize(
    "rows_max_bytes,expected_num_rows_items,expected_truncated",
    [
        # the rows are kept as is
        (5_000, DEFAULT_NUM_ROWS, False),
        (2_000, DEFAULT_NUM_ROWS, False),
        # some rows have been removed at the end, the rest is kept as is
        (1_500, 10, True),
        (1_000, 7, True),
        (900, 6, True),
        # all the row above the limit have been removed, and the rest have been truncated
        (500, DEFAULT_ROWS_MIN_NUMBER, True),
        (100, DEFAULT_ROWS_MIN_NUMBER, True),
    ],
)
# all other things being equal, decreasing the maximum size of the rows decreases the number of rows after truncation
def test_create_truncated_row_items(
    rows_max_bytes: int,
    expected_num_rows_items: int,
    expected_truncated: bool,
) -> None:
    rows = [
        {f"c{i}": "a" * DEFAULT_CELL_BYTES for i in range(DEFAULT_NUM_CELLS)} for row_idx in range(DEFAULT_NUM_ROWS)
    ]
    (truncated_row_items, truncated) = create_truncated_row_items(
        rows=rows,
        rows_max_bytes=rows_max_bytes,
        min_cell_bytes=DEFAULT_MIN_CELL_BYTES,
        rows_min_number=DEFAULT_ROWS_MIN_NUMBER,
        columns_to_keep_untruncated=[],
    )
    assert len(truncated_row_items) == expected_num_rows_items
    assert truncated == expected_truncated
