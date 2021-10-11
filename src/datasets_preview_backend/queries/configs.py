from typing import List, TypedDict

from datasets_preview_backend.dataset_entries import get_dataset_entry


class ConfigItem(TypedDict):
    dataset: str
    config: str


class ConfigsContent(TypedDict):
    configs: List[ConfigItem]


def get_configs(dataset: str) -> ConfigsContent:
    dataset_entry = get_dataset_entry(dataset=dataset)
    return {
        "configs": [
            {"dataset": dataset_entry["dataset"], "config": config_entry["config"]}
            for config_entry in dataset_entry["configs"]
        ]
    }
