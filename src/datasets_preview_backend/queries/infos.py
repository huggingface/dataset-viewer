# TODO: deprecate?
from typing import Any, Dict, List, Optional, TypedDict

from datasets_preview_backend.models.config import filter_configs
from datasets_preview_backend.models.dataset import get_dataset


class InfoItem(TypedDict):
    dataset: str
    config: str
    info: Dict[str, Any]


class InfosContent(TypedDict):
    infos: List[InfoItem]


def get_infos(dataset_name: str, config_name: Optional[str] = None) -> InfosContent:
    dataset = get_dataset(dataset_name=dataset_name)

    return {
        "infos": [
            {"dataset": dataset_name, "config": config["config_name"], "info": config["info"]}
            for config in filter_configs(dataset["configs"], config_name)
        ]
    }
