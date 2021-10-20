import logging
from typing import List, Optional, TypedDict

from datasets_preview_backend.models.column import JsonColumn
from datasets_preview_backend.models.config import filter_configs
from datasets_preview_backend.models.dataset import get_dataset
from datasets_preview_backend.models.row import Row
from datasets_preview_backend.models.split import filter_splits

logger = logging.getLogger(__name__)


class ColumnItem(TypedDict):
    dataset: str
    config: str
    split: str
    column: JsonColumn


class RowItem(TypedDict):
    dataset: str
    config: str
    split: str
    row: Row


class RowsContent(TypedDict):
    columns: List[ColumnItem]
    rows: List[RowItem]


def get_rows(dataset_name: str, config_name: Optional[str] = None, split_name: Optional[str] = None) -> RowsContent:
    if config_name is None:
        # split is ignored if config is not passed
        logger.debug("split argument is ignored since config is not provided")
        split_name = None

    dataset = get_dataset(dataset_name=dataset_name)

    configs = filter_configs(dataset["configs"], config_name)

    return {
        "columns": [
            {
                "dataset": dataset_name,
                "config": config["config_name"],
                "split": split["split_name"],
                "column": column.to_json(),
            }
            for config in configs
            for split in filter_splits(config["splits"], split_name)
            for column in split["columns"]
        ],
        "rows": [
            {
                "dataset": dataset_name,
                "config": config["config_name"],
                "split": split["split_name"],
                "row": row,
            }
            for config in configs
            for split in filter_splits(config["splits"], split_name)
            for row in split["rows"]
        ],
    }
