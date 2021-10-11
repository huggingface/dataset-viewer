from typing import List, Optional, TypedDict

from datasets_preview_backend.dataset_entries import (
    filter_config_entries,
    get_dataset_entry,
)


class SplitItem(TypedDict):
    dataset: str
    config: str
    split: str


class SplitsContent(TypedDict):
    splits: List[SplitItem]


def get_splits(dataset: str, config: Optional[str] = None) -> SplitsContent:
    dataset_entry = get_dataset_entry(dataset=dataset)

    return {
        "splits": [
            {"dataset": dataset, "config": config_entry["config"], "split": split_entry["split"]}
            for config_entry in filter_config_entries(dataset_entry["configs"], config)
            for split_entry in config_entry["splits"]
        ]
    }
