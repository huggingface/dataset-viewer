# TODO: deprecate?
from typing import List, TypedDict

from datasets_preview_backend.models.dataset import get_dataset


class ConfigItem(TypedDict):
    dataset: str
    config: str


class ConfigsContent(TypedDict):
    configs: List[ConfigItem]


def get_configs(dataset_name: str) -> ConfigsContent:
    dataset = get_dataset(dataset_name=dataset_name)
    return {
        "configs": [
            {"dataset": dataset["dataset_name"], "config": config["config_name"]} for config in dataset["configs"]
        ]
    }
