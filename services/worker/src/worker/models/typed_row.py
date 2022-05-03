import logging
from typing import List, Optional, Tuple

from datasets import DatasetInfo
from libutils.exceptions import Status400Error

from worker.models.column import Column, get_columns
from worker.models.row import Row, get_rows

logger = logging.getLogger(__name__)


def get_typed_row(
    dataset_name: str, config_name: str, split_name: str, row: Row, row_idx: int, columns: List[Column]
) -> Row:
    return {
        column.name: column.get_cell_value(dataset_name, config_name, split_name, row_idx, row[column.name])
        for column in columns
    }


def get_typed_rows(
    dataset_name: str,
    config_name: str,
    split_name: str,
    rows: List[Row],
    columns: List[Column],
) -> List[Row]:
    return [get_typed_row(dataset_name, config_name, split_name, row, idx, columns) for idx, row in enumerate(rows)]


def get_typed_rows_and_columns(
    dataset_name: str,
    config_name: str,
    split_name: str,
    info: DatasetInfo,
    hf_token: Optional[str] = None,
    fallback: Optional[bool] = False,
    rows_max_number: Optional[int] = None,
) -> Tuple[List[Row], List[Column]]:
    try:
        try:
            rows = get_rows(dataset_name, config_name, split_name, hf_token, True, rows_max_number)
        except Exception:
            if fallback:
                rows = get_rows(dataset_name, config_name, split_name, hf_token, False, rows_max_number)
            else:
                raise
    except Exception as err:
        raise Status400Error("Cannot get the first rows for the split.", err) from err

    columns = get_columns(info, rows)
    typed_rows = get_typed_rows(dataset_name, config_name, split_name, rows, columns)
    return typed_rows, columns
