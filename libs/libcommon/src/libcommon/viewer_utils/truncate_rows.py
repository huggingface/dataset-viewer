# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.


from libcommon.dtos import Row, RowItem
from libcommon.utils import SmallerThanMaxBytesError, get_json_size, serialize_and_truncate


def to_row_item(row_idx: int, row: Row) -> RowItem:
    return {
        "row_idx": row_idx,
        "row": row,
        "truncated_cells": [],
    }


def truncate_row_item(row_item: RowItem, min_cell_bytes: int, columns_to_keep_untruncated: list[str]) -> RowItem:
    """
    Truncate all the cells of a row item to min_cell_bytes, and return the row item.

    The row item is mutated, and the cells are replaced by their JSON serialization, truncated to min_cell_bytes.
    The names of the truncated cells are listed in row_item["truncated_cells"].

    Args:
        row_item ([`RowItem`]): the row item to truncate.
        min_cell_bytes (`int`): the minimum number of bytes for a cell. If a cell has less than this number of bytes,
            it is not truncated. If it has more, it is truncated to this number of bytes.
            The size of a cell is computed as the size of its JSON serialization using orjson_dumps().
        columns_to_keep_untruncated (`list[str]`): the list of columns to keep untruncated.

    Returns:
        [`RowItem`]: the same row item, mutated, with all the cells truncated to min_cell_bytes.
    """
    for column_name, cell in row_item["row"].items():
        if column_name in columns_to_keep_untruncated:
            # we keep the cell untouched
            continue
        try:
            truncated_serialized_cell = serialize_and_truncate(obj=cell, max_bytes=min_cell_bytes)
            row_item["row"][column_name] = truncated_serialized_cell
            if column_name not in row_item["truncated_cells"]:
                row_item["truncated_cells"].append(column_name)
        except SmallerThanMaxBytesError:
            # the cell serialization is smaller than min_cell_bytes, we keep it untouched
            continue
    return row_item


def truncate_row_items_cells(
    row_items: list[RowItem], min_cell_bytes: int, rows_max_bytes: int, columns_to_keep_untruncated: list[str]
) -> list[RowItem]:
    """
    Truncate the cells of a list of row items to fit within a maximum number of bytes.

    The row items are mutated, and the cells are replaced by their JSON serialization, truncated to min_cell_bytes.
    The names of the truncated cells are listed in row_item["truncated_cells"].

    The rows are truncated in reverse order, starting from the last row, until the sum of the rows is under the
    rows_max_bytes threshold, or until the first row.

    Note that only the content of the cells (row_item["row"][column_name]) is truncated, while the other fields
    like row_item["row_idx"] and row_item["truncated_cells"] account for the size of the response. This means that
    the size of the response might be greater than rows_max_bytes, even if all the rows have been "truncated".

    Args:
        row_items (`list[RowItem]`): the row items to truncate.
        min_cell_bytes (`int`): the minimum number of bytes for a cell. If a cell has less than this number of bytes,
            it is not truncated. If it has more, it is serialized to JSON and truncated to this number of bytes.
            The size of a cell is computed as the size of its JSON serialization using orjson_dumps().
        rows_max_bytes (`int`): the maximum number of bytes of the rows JSON serialization, after truncation of the last ones.
            The size accounts for the comma separators between rows.
        columns_to_keep_untruncated (`list[str]`): the list of columns to keep untruncated.

    Returns:
        list[`RowItem`]: the same row items, mutated.
    """
    # compute the current size
    rows_bytes = get_json_size(row_items)

    # Loop backwards, so that the last rows are truncated first
    for row_item in reversed(row_items):
        if rows_bytes < rows_max_bytes:
            break
        previous_size = get_json_size(row_item)
        row_item = truncate_row_item(
            row_item=row_item, min_cell_bytes=min_cell_bytes, columns_to_keep_untruncated=columns_to_keep_untruncated
        )
        new_size = get_json_size(row_item)
        rows_bytes += new_size - previous_size
    return row_items


COMMA_SIZE = 1  # the comma "," is encoded with one byte in utf-8
BRACKET_SIZE = 1  # the brackets "[" and "]" are encoded with one byte in utf-8


def create_truncated_row_items(
    rows: list[Row],
    min_cell_bytes: int,
    rows_max_bytes: int,
    rows_min_number: int,
    columns_to_keep_untruncated: list[str],
) -> tuple[list[RowItem], bool]:
    """
    Truncate the rows to fit within the restrictions, and prepare them as RowItems.

    The rows are removed in reverse order, starting from the last row, until the sum of the rows is under the
    rows_max_bytes threshold, or until the minimal number of rows has been reached.

    If the sum of the rows is still above the rows_max_bytes threshold, the cells inside the remaining rows are
    serialized to JSON and truncated to min_cell_bytes. This process starts from the last remaining row, and goes
    backwards until the sum of the rows is under the rows_max_bytes threshold, or until the first row.

    Note that the size of the response might be greater than rows_max_bytes, even if all the rows have been "truncated".
    This is because the size of the response accounts for the other fields like row_item["row_idx"] and
    row_item["truncated_cells"], while only the row_item["cell"] contents are truncated.

    Args:
        rows (`list[Row]`): the rows to truncate.
        min_cell_bytes (`int`): the minimum number of bytes for a cell. If a cell has less than this number of bytes,
            it is not truncated. If it has more, it is serialized to JSON and truncated to this number of bytes.
            The size of a cell is computed as the size of its JSON serialization using orjson_dumps().
        rows_max_bytes (`int`): the maximum number of bytes of the rows JSON serialization, after truncation of the last ones.
            The size accounts for the comma separators between rows.
        rows_min_number (`int`): the minimum number of rows to keep.
        columns_to_keep_untruncated (`list[str]`): the list of columns to keep untruncated.

    Returns:
        `tuple[list[RowItem], bool]`:
        - `list[RowItem]`: the list of row items.
        - `bool`: a boolean indicating whether the rows have been truncated. False if the rows have been kept untouched,
            True if they have been deleted or truncated.
    """
    row_items = []
    rows_bytes = 2 * BRACKET_SIZE

    # two restrictions must be enforced:
    # - at least rows_min_number rows
    # - at most rows_max_bytes bytes. Note that it's the limit to the sum of the rows sizes. The JSON response size
    #   will be greater, due to the other fields (row_idx, truncated_cells, features, etc.).
    # To enforce this:
    # 1. first get the first rows_min_number rows
    for row_idx, row in enumerate(rows[:rows_min_number]):
        row_item = to_row_item(row_idx=row_idx, row=row)
        rows_bytes += get_json_size(row_item) + COMMA_SIZE
        row_items.append(row_item)

    # 2. if the total is over the bytes limit, truncate the values, iterating backwards starting
    # from the last rows, until getting under the threshold
    # caveat: the truncation might not be enough to get under the threshold if:
    # - the number of columns is too high
    # - rows_max_bytes is too low (or even negative)
    if rows_bytes >= rows_max_bytes:
        # logging.debug(
        #     f"the size of the first {rows_min_number} rows ({rows_bytes}) is above the max number of bytes"
        #     f" ({rows_max_bytes}), they will be truncated"
        # )
        truncated_row_items = truncate_row_items_cells(
            row_items=row_items,
            min_cell_bytes=min_cell_bytes,
            rows_max_bytes=rows_max_bytes,
            columns_to_keep_untruncated=columns_to_keep_untruncated,
        )
        return truncated_row_items, len(truncated_row_items) < len(rows)

    # 3. else: add the remaining rows until the end, or until the bytes threshold
    for idx, row in enumerate(rows[rows_min_number:]):
        row_idx = rows_min_number + idx
        row_item = to_row_item(row_idx=row_idx, row=row)
        rows_bytes += get_json_size(row_item) + COMMA_SIZE
        if rows_bytes >= rows_max_bytes:
            # logging.debug(
            #     f"the rows in the split have been truncated to {row_idx} row(s) to keep the size"
            #     f" ({rows_bytes}) under the limit ({rows_max_bytes})"
            # )
            break
        row_items.append(row_item)
    return row_items, len(row_items) < len(rows)
