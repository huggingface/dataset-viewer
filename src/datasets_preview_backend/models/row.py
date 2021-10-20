import logging
import re
from typing import Any, Dict, List, Tuple, Union

from datasets import IterableDataset, load_dataset

from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.constants import FORCE_REDOWNLOAD
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.models.column import Column

logger = logging.getLogger(__name__)


Row = Dict[str, Any]


def get_rows(dataset: str, config: str, split: str) -> List[Row]:
    num_rows = EXTRACT_ROWS_LIMIT
    try:
        iterable_dataset = load_dataset(
            dataset, name=config, split=split, streaming=True, download_mode=FORCE_REDOWNLOAD  # type: ignore
        )
        if not isinstance(iterable_dataset, IterableDataset):
            raise TypeError("load_dataset should return an IterableDataset")
        rows = list(iterable_dataset.take(num_rows))
    except FileNotFoundError as err:
        raise Status404Error("The split for the dataset config could not be found.", err)
    except NotImplementedError as err:
        # TODO: check what has changed once https://github.com/huggingface/datasets/pull/2662 is merged
        try:
            regex = re.compile(r"Extraction protocol for file at .*?((\.\w+)?\.\w+)* is not implemented yet")
            match = regex.match(str(err))
            if match is None:
                raise Exception("No match")
            extension = match.group(1)
        except Exception:
            raise Status400Error("The rows could not be extracted from the split of the dataset config.", err)
        else:
            raise Status400Error(
                "The rows could not be extracted from the split of the dataset config because extension"
                f" {extension} is not supported.",
                err,
            )
    except ValueError as err:
        if (
            str(err).startswith(f"BuilderConfig {config} not found.")
            or str(err).startswith("Config name is missing.")
            or str(err).startswith("Bad split")
        ):
            raise Status404Error("The dataset config could not be found.", err)
        else:
            raise Status400Error("The rows could not be extracted from the split of the dataset config.", err)
    except Exception as err:
        raise Status400Error("The rows could not be extracted from the split of the dataset config.", err)

    if len(rows) != num_rows:
        logger.info(
            f"could not read all the required rows ({len(rows)} / {num_rows}) from dataset {dataset} -"
            f" {config} - {split}"
        )

    return rows


def generate_typed_row(dataset: str, config: str, split: str, row: Row, row_idx: int, columns: List[Column]) -> Row:
    return {
        column.name: column.get_cell_value(dataset, config, split, row_idx, row[column.name]) for column in columns
    }


def get_columns_from_row(row: Row) -> List[Column]:
    # TODO: try to guess the column type from the values instead of just using the default type?
    return [Column(name, None) for name in row.keys()]


def check_columns(columns: List[Column], row: Row) -> None:
    if len(row) != len(columns):
        raise Status400Error("number of columns in features and row is different")
    column_names = [column.name for column in columns]
    row_names = list(row.keys())
    if len(column_names) != len(set(column_names)):
        raise Status400Error("duplicate feature names")
    if len(row_names) != len(set(row_names)):
        raise Status400Error("duplicate column names in row")
    if len(set(column_names) - set(row_names)) != 0:
        raise Status400Error("column names mismatch between features and row")


def get_rows_and_columns(
    dataset_name: str, config_name: str, split_name: str, columns_or_none: Union[List[Column], None]
) -> Tuple[List[Row], List[Column]]:
    rows = get_rows(dataset_name, config_name, split_name)
    if not rows:
        return [], [] if columns_or_none is None else columns_or_none
    columns = get_columns_from_row(rows[0]) if columns_or_none is None else columns_or_none
    check_columns(columns, rows[0])
    typed_rows = [
        generate_typed_row(dataset_name, config_name, split_name, row, row_idx, columns)
        for row_idx, row in enumerate(rows)
    ]
    return typed_rows, columns
