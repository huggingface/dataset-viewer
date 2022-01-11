from typing import List, Optional, TypedDict

from datasets import get_dataset_split_names

from datasets_preview_backend.constants import FORCE_REDOWNLOAD
from datasets_preview_backend.exceptions import Status400Error
from datasets_preview_backend.models.column import Column
from datasets_preview_backend.models.info import Info
from datasets_preview_backend.models.row import Row
from datasets_preview_backend.models.typed_row import get_typed_rows_and_columns


class Split(TypedDict):
    split_name: str
    rows: List[Row]
    columns: List[Column]


def get_split(
    dataset_name: str,
    config_name: str,
    split_name: str,
    info: Info,
    hf_token: Optional[str] = None,
    max_size_fallback: Optional[int] = None,
) -> Split:
    fallback = max_size_fallback is not None and "size_in_bytes" in info and info["size_in_bytes"] < max_size_fallback
    typed_rows, columns = get_typed_rows_and_columns(dataset_name, config_name, split_name, info, hf_token, fallback)
    return {"split_name": split_name, "rows": typed_rows, "columns": columns}


def get_split_names(dataset_name: str, config_name: str, hf_token: Optional[str] = None) -> List[str]:
    try:
        split_names: List[str] = get_dataset_split_names(
            dataset_name, config_name, download_mode=FORCE_REDOWNLOAD, use_auth_token=hf_token  # type: ignore
        )
    except Exception as err:
        raise Status400Error("Cannot get the split names for the dataset config.", err)
    return split_names


def get_splits(
    dataset_name: str,
    config_name: str,
    info: Info,
    hf_token: Optional[str] = None,
    max_size_fallback: Optional[int] = None,
) -> List[Split]:
    return [
        get_split(dataset_name, config_name, split_name, info, hf_token, max_size_fallback)
        for split_name in get_split_names(dataset_name, config_name, hf_token)
    ]
