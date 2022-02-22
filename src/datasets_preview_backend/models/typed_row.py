from typing import List, Optional, Tuple

from datasets import DatasetInfo

from datasets_preview_backend.models.column import Column, get_columns
from datasets_preview_backend.models.row import (
    Row,
    get_rows,
    get_rows_without_streaming,
)


def get_typed_row(
    dataset_name: str, config_name: str, split_name: str, row: Row, row_idx: int, columns: List[Column]
) -> Row:
    return {
        column.name: column.get_cell_value(dataset_name, config_name, split_name, row_idx, row[column.name])
        for column in columns
    }


def get_typed_rows(
    dataset_name: str, config_name: str, split_name: str, rows: List[Row], columns: List[Column]
) -> List[Row]:
    return [
        get_typed_row(dataset_name, config_name, split_name, row, row_idx, columns) for row_idx, row in enumerate(rows)
    ]


def get_typed_rows_and_columns(
    dataset_name: str,
    config_name: str,
    split_name: str,
    info: DatasetInfo,
    hf_token: Optional[str] = None,
    fallback: Optional[bool] = False,
) -> Tuple[List[Row], List[Column]]:
    try:
        rows = get_rows(dataset_name, config_name, split_name, hf_token)
    except Exception:
        if fallback:
            rows = get_rows_without_streaming(dataset_name, config_name, split_name, hf_token)
        else:
            raise
    columns = get_columns(info, rows)
    typed_rows = get_typed_rows(dataset_name, config_name, split_name, rows, columns)
    return typed_rows, columns
