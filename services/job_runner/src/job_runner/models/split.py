import logging
import sys
from typing import Any, List, Optional

from libutils.utils import orjson_dumps
from libutils.types import RowItem, ColumnItem, RowsResponse, Split

from job_runner.models._guard import guard_blocked_datasets
from job_runner.models.column import Column
from job_runner.models.constants import DEFAULT_MIN_CELL_BYTES
from job_runner.models.info import get_info
from job_runner.models.row import Row
from job_runner.models.typed_row import get_typed_rows_and_columns

logger = logging.getLogger(__name__)


def get_size_in_bytes(obj: Any):
    return sys.getsizeof(orjson_dumps(obj))
    # ^^ every row is transformed here in a string, because it corresponds to
    # the size the row will contribute in the JSON response to /rows endpoint.
    # The size of the string is measured in bytes.
    # An alternative would have been to look at the memory consumption (pympler) but it's
    # less related to what matters here (size of the JSON, number of characters in the
    # dataset viewer table on the hub)


def truncate_cell(cell: Any, min_cell_bytes: int) -> str:
    return orjson_dumps(cell)[:min_cell_bytes].decode("utf8", "ignore")


# Mutates row_item, and returns it anyway
def truncate_row_item(row_item: RowItem) -> RowItem:
    min_cell_bytes = DEFAULT_MIN_CELL_BYTES
    row = {}
    for column_name, cell in row_item["row"].items():
        # for now: all the cells, but the smallest ones, are truncated
        cell_bytes = get_size_in_bytes(cell)
        if cell_bytes > min_cell_bytes:
            row_item["truncated_cells"].append(column_name)
            row[column_name] = truncate_cell(cell, min_cell_bytes)
        else:
            row[column_name] = cell
    row_item["row"] = row
    return row_item


# Mutates row_items, and returns them anyway
def truncate_row_items(row_items: List[RowItem], rows_max_bytes: int) -> List[RowItem]:
    # compute the current size
    rows_bytes = sum(get_size_in_bytes(row_item) for row_item in row_items)

    # Loop backwards, so that the last rows are truncated first
    for row_item in reversed(row_items):
        if rows_bytes < rows_max_bytes:
            break
        previous_size = get_size_in_bytes(row_item)
        row_item = truncate_row_item(row_item)
        new_size = get_size_in_bytes(row_item)
        rows_bytes += new_size - previous_size
        row_idx = row_item["row_idx"]
        logger.debug(f"the size of the rows is now ({rows_bytes}) after truncating row idx={row_idx}")
    return row_items


def to_row_item(dataset_name: str, config_name: str, split_name: str, row_idx: int, row: Row) -> RowItem:
    return {
        "dataset": dataset_name,
        "config": config_name,
        "split": split_name,
        "row_idx": row_idx,
        "row": row,
        "truncated_cells": [],
    }


def to_column_item(
    dataset_name: str, config_name: str, split_name: str, column_idx: int, column: Column
) -> ColumnItem:
    return {
        "dataset": dataset_name,
        "config": config_name,
        "split": split_name,
        "column_idx": column_idx,
        "column": column.as_dict(),
    }


def create_truncated_row_items(
    dataset_name: str,
    config_name: str,
    split_name: str,
    rows: List[Row],
    rows_max_bytes: Optional[int] = None,
    rows_min_number: Optional[int] = None,
) -> List[RowItem]:
    row_items = []
    rows_bytes = 0
    if rows_min_number is None:
        rows_min_number = 0
    else:
        logger.debug(f"min number of rows in the response: '{rows_min_number}'")
    if rows_max_bytes is not None:
        logger.debug(f"max number of bytes in the response: '{rows_max_bytes}'")

    # two restrictions must be enforced:
    # - at least rows_min_number rows
    # - at most rows_max_bytes bytes
    # To enforce this:
    # 1. first get the first rows_min_number rows
    for row_idx, row in enumerate(rows[:rows_min_number]):
        row_item = to_row_item(dataset_name, config_name, split_name, row_idx, row)
        if rows_max_bytes is not None:
            rows_bytes += get_size_in_bytes(row_item)
        row_items.append(row_item)

    # 2. if the total is over the bytes limit, truncate the values, iterating backwards starting
    # from the last rows, until getting under the threshold
    if rows_max_bytes is not None and rows_bytes >= rows_max_bytes:
        logger.debug(
            f"the size of the first {rows_min_number} rows ({rows_bytes}) is above the max number of bytes"
            f" ({rows_max_bytes}), they will be truncated"
        )
        return truncate_row_items(row_items, rows_max_bytes)

    # 3. else: add the remaining rows until the end, or until the bytes threshold
    for idx, row in enumerate(rows[rows_min_number:]):
        row_idx = rows_min_number + idx
        row_item = to_row_item(dataset_name, config_name, split_name, row_idx, row)
        if rows_max_bytes is not None:
            rows_bytes += get_size_in_bytes(row_item)
            if rows_bytes >= rows_max_bytes:
                logger.debug(
                    f"the rows in the split have been truncated to {row_idx} row(s) to keep the size"
                    f" ({rows_bytes}) under the limit ({rows_max_bytes})"
                )
                break
        row_items.append(row_item)
    return row_items


def get_split(
    dataset_name: str,
    config_name: str,
    split_name: str,
    hf_token: Optional[str] = None,
    max_size_fallback: Optional[int] = None,
    rows_max_bytes: Optional[int] = None,
    rows_max_number: Optional[int] = None,
    rows_min_number: Optional[int] = None,
) -> Split:
    logger.info(f"get split '{split_name}' for config '{config_name}' of dataset '{dataset_name}'")
    guard_blocked_datasets(dataset_name)
    info = get_info(dataset_name, config_name, hf_token)
    fallback = (
        max_size_fallback is not None and info.size_in_bytes is not None and info.size_in_bytes < max_size_fallback
    )
    typed_rows, columns = get_typed_rows_and_columns(
        dataset_name, config_name, split_name, info, hf_token, fallback, rows_max_number
    )
    row_items = create_truncated_row_items(
        dataset_name, config_name, split_name, typed_rows, rows_max_bytes, rows_min_number
    )
    rows_response: RowsResponse = {
        "columns": [
            to_column_item(dataset_name, config_name, split_name, column_idx, column)
            for column_idx, column in enumerate(columns)
        ],
        "rows": row_items,
    }
    try:
        if info.splits is None:
            raise Exception("no splits in info")
        num_bytes = info.splits[split_name].num_bytes
        num_examples = info.splits[split_name].num_examples
    except Exception:
        num_bytes = None
        num_examples = None
    return {
        "split_name": split_name,
        "rows_response": rows_response,
        "num_bytes": num_bytes,
        "num_examples": num_examples,
    }
