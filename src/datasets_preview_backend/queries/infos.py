from typing import Any, Dict, List, Optional, TypedDict

from datasets_preview_backend.dataset_entries import (
    filter_config_entries,
    get_dataset_entry,
)


class InfoItem(TypedDict):
    dataset: str
    config: str
    info: Dict[str, Any]


class InfosContent(TypedDict):
    infos: List[InfoItem]


def get_infos(dataset: str, config: Optional[str] = None) -> InfosContent:
    dataset_entry = get_dataset_entry(dataset=dataset)

    return {
        "infos": [
            {"dataset": dataset, "config": config_entry["config"], "info": config_entry["info"]}
            for config_entry in filter_config_entries(dataset_entry["configs"], config)
        ]
    }
