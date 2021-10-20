from typing import List, Optional, TypedDict, Union

from datasets import get_dataset_split_names

from datasets_preview_backend.constants import FORCE_REDOWNLOAD
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.models.column import Column
from datasets_preview_backend.models.row import Row, get_rows_and_columns


class Split(TypedDict):
    split_name: str
    rows: List[Row]
    columns: List[Column]


def get_split(
    dataset_name: str, config_name: str, split_name: str, columns_or_none: Union[List[Column], None]
) -> Split:
    rows, columns = get_rows_and_columns(dataset_name, config_name, split_name, columns_or_none)
    return {"split_name": split_name, "rows": rows, "columns": columns}


def get_split_names(dataset_name: str, config_name: str) -> List[str]:
    try:
        split_names: List[str] = get_dataset_split_names(
            dataset_name, config_name, download_mode=FORCE_REDOWNLOAD  # type: ignore
        )
    except FileNotFoundError as err:
        raise Status404Error("The dataset config could not be found.", err)
    except ValueError as err:
        if str(err).startswith(f"BuilderConfig {config_name} not found."):
            raise Status404Error("The dataset config could not be found.", err)
        else:
            raise Status400Error("The split names could not be parsed from the dataset config.", err)
    except Exception as err:
        raise Status400Error("The split names could not be parsed from the dataset config.", err)
    return split_names


def get_splits(dataset_name: str, config_name: str, columns_or_none: Union[List[Column], None]) -> List[Split]:
    return [
        get_split(dataset_name, config_name, split_name, columns_or_none)
        for split_name in get_split_names(dataset_name, config_name)
    ]


def filter_splits(splits: List[Split], split_name: Optional[str] = None) -> List[Split]:
    if split_name is not None:
        if not isinstance(split_name, str):
            raise TypeError("split_name argument should be a string")
        splits = [split for split in splits if split["split_name"] == split_name]
        if not splits:
            raise Status404Error("split_name not found in config")
    return splits
