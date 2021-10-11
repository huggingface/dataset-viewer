import logging
from typing import Any, List, Optional, TypedDict

from datasets_preview_backend.dataset_entries import (
    filter_config_entries,
    filter_split_entries,
    get_dataset_entry,
)

logger = logging.getLogger(__name__)


class Feature(TypedDict):
    name: str
    content: Any


class FeatureItem(TypedDict):
    dataset: str
    config: str
    feature: Feature


class RowItem(TypedDict):
    dataset: str
    config: str
    split: str
    row: Any


class RowsContent(TypedDict):
    features: List[FeatureItem]
    rows: List[RowItem]


def get_rows(dataset: str, config: Optional[str] = None, split: Optional[str] = None) -> RowsContent:
    if config is None:
        # split is ignored if config is not passed
        logger.debug("split argument is ignored since config is not provided")
        split = None

    dataset_entry = get_dataset_entry(dataset=dataset)

    config_entries = filter_config_entries(dataset_entry["configs"], config)

    return {
        "features": [
            {"dataset": dataset, "config": config_entry["config"], "feature": feature}
            for config_entry in config_entries
            for feature in config_entry["features"]
        ],
        "rows": [
            {"dataset": dataset, "config": config_entry["config"], "split": split_entry["split"], "row": row}
            for config_entry in config_entries
            for split_entry in filter_split_entries(config_entry["splits"], split)
            for row in split_entry["rows"]
        ],
    }
