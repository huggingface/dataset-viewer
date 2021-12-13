from typing import List, TypedDict

from datasets import get_dataset_split_names

from datasets_preview_backend.constants import FORCE_REDOWNLOAD
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.models.column import Column
from datasets_preview_backend.models.info import Info
from datasets_preview_backend.models.row import Row
from datasets_preview_backend.models.typed_row import get_typed_rows_and_columns


class Split(TypedDict):
    split_name: str
    rows: List[Row]
    columns: List[Column]


def get_split(dataset_name: str, config_name: str, split_name: str, info: Info) -> Split:
    typed_rows, columns = get_typed_rows_and_columns(dataset_name, config_name, split_name, info)
    return {"split_name": split_name, "rows": typed_rows, "columns": columns}


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


def get_splits(dataset_name: str, config_name: str, info: Info) -> List[Split]:
    return [
        get_split(dataset_name, config_name, split_name, info)
        for split_name in get_split_names(dataset_name, config_name)
    ]
