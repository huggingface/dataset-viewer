import time
from typing import List, TypedDict

from datasets_preview_backend.dataset_entries import (
    get_dataset_cache_status,
    get_refreshed_dataset_names,
)


class DatasetsByStatus(TypedDict):
    valid: List[str]
    error: List[str]
    cache_miss: List[str]
    created_at: str


def get_valid_datasets() -> DatasetsByStatus:
    statuses = [get_dataset_cache_status(dataset) for dataset in get_refreshed_dataset_names()]
    return {
        "valid": [status["dataset"] for status in statuses if status["status"] == "valid"],
        "error": [status["dataset"] for status in statuses if status["status"] == "error"],
        "cache_miss": [status["dataset"] for status in statuses if status["status"] == "cache_miss"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
