import logging
import sys
from typing import List, Optional, Tuple

from datasets import DatasetInfo

from datasets_preview_backend.config import ROWS_MAX_BYTES
from datasets_preview_backend.exceptions import Status400Error
from datasets_preview_backend.models.column import Column, get_columns
from datasets_preview_backend.models.row import Row, get_rows
from datasets_preview_backend.utils import orjson_dumps

logger = logging.getLogger(__name__)


def get_typed_row(
    dataset_name: str, config_name: str, split_name: str, row: Row, row_idx: int, columns: List[Column]
) -> Row:
    return {
        column.name: column.get_cell_value(dataset_name, config_name, split_name, row_idx, row[column.name])
        for column in columns
    }


def get_size_in_bytes(row: Row):
    return sys.getsizeof(orjson_dumps(row))
    # ^^ every row is transformed here in a string, because it corresponds to
    # the size the row will contribute in the JSON response to /rows endpoint.
    # The size of the string is measured in bytes.
    # An alternative would have been to look at the memory consumption (pympler) but it's
    # less related to what matters here (size of the JSON, number of characters in the
    # dataset viewer table on the hub)


def get_typed_rows(
    dataset_name: str,
    config_name: str,
    split_name: str,
    rows: List[Row],
    columns: List[Column],
    rows_max_bytes: Optional[int] = None,
) -> List[Row]:
    typed_rows = []
    bytes = 0
    for idx, row in enumerate(rows):
        typed_row = get_typed_row(dataset_name, config_name, split_name, row, idx, columns)
        if ROWS_MAX_BYTES is not None:
            bytes += get_size_in_bytes(typed_row)
            if bytes >= rows_max_bytes:
                logger.debug(
                    f"the rows in the split have been truncated to {idx} row(s) to keep the size ({bytes}) under the"
                    f" limit ({rows_max_bytes})"
                )
                break
        typed_rows.append(typed_row)
    return typed_rows


def get_typed_rows_and_columns(
    dataset_name: str,
    config_name: str,
    split_name: str,
    info: DatasetInfo,
    hf_token: Optional[str] = None,
    fallback: Optional[bool] = False,
) -> Tuple[List[Row], List[Column]]:
    try:
        try:
            rows = get_rows(dataset_name, config_name, split_name, hf_token, streaming=True)
        except Exception:
            if fallback:
                rows = get_rows(dataset_name, config_name, split_name, hf_token, streaming=False)
            else:
                raise
    except Exception as err:
        raise Status400Error("Cannot get the first rows for the split.", err) from err

    columns = get_columns(info, rows)
    typed_rows = get_typed_rows(dataset_name, config_name, split_name, rows, columns, ROWS_MAX_BYTES)
    return typed_rows, columns
