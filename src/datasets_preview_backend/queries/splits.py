from typing import List, Optional, TypedDict

from datasets_preview_backend.models.config import filter_configs
from datasets_preview_backend.models.dataset import get_dataset


class SplitItem(TypedDict):
    dataset: str
    config: str
    split: str


class SplitsContent(TypedDict):
    splits: List[SplitItem]


def get_splits(dataset_name: str, config_name: Optional[str] = None) -> SplitsContent:
    dataset = get_dataset(dataset_name=dataset_name)

    return {
        "splits": [
            {"dataset": dataset_name, "config": config["config_name"], "split": split["split_name"]}
            for config in filter_configs(dataset["configs"], config_name)
            for split in config["splits"]
        ]
    }
