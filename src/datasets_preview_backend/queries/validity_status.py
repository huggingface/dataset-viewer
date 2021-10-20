import time
from typing import List, TypedDict

from datasets_preview_backend.models.dataset import get_dataset_cache_status
from datasets_preview_backend.models.hf_dataset import get_refreshed_hf_dataset_names


class DatasetsByStatus(TypedDict):
    valid: List[str]
    error: List[str]
    cache_miss: List[str]
    created_at: str


def get_valid_datasets() -> DatasetsByStatus:
    statuses = [get_dataset_cache_status(dataset_name) for dataset_name in get_refreshed_hf_dataset_names()]
    return {
        "valid": [status["dataset_name"] for status in statuses if status["status"] == "valid"],
        "error": [status["dataset_name"] for status in statuses if status["status"] == "error"],
        "cache_miss": [status["dataset_name"] for status in statuses if status["status"] == "cache_miss"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
