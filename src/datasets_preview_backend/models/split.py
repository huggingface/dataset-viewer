import logging
from typing import List, Optional, TypedDict

from datasets_preview_backend.models._guard import guard_blocked_datasets
from datasets_preview_backend.models.column import Column
from datasets_preview_backend.models.info import get_info
from datasets_preview_backend.models.row import Row
from datasets_preview_backend.models.typed_row import get_typed_rows_and_columns

logger = logging.getLogger(__name__)


class Split(TypedDict):
    split_name: str
    rows: List[Row]
    columns: List[Column]


def get_split(
    dataset_name: str,
    config_name: str,
    split_name: str,
    hf_token: Optional[str] = None,
    max_size_fallback: Optional[int] = None,
) -> Split:
    logger.info(f"get split '{split_name}' for config '{config_name}' of dataset '{dataset_name}'")
    guard_blocked_datasets(dataset_name)
    info = get_info(dataset_name, config_name, hf_token)
    fallback = (
        max_size_fallback is not None
        and "size_in_bytes" in info
        and isinstance(info["size_in_bytes"], int)
        and info["size_in_bytes"] < max_size_fallback
    )
    typed_rows, columns = get_typed_rows_and_columns(dataset_name, config_name, split_name, info, hf_token, fallback)
    try:
        num_bytes = info["splits"][split_name]["num_bytes"]
        num_examples = info["splits"][split_name]["num_examples"]
    except Exception:
        num_bytes = None
        num_examples = None
    return {
        "split_name": split_name,
        "rows": typed_rows,
        "columns": columns,
        "num_bytes": num_bytes,
        "num_examples": num_examples,
    }
