import time
from typing import TypedDict

from datasets_preview_backend.models.dataset import get_dataset_cache_status
from datasets_preview_backend.models.hf_dataset import get_refreshed_hf_dataset_names


class CacheStats(TypedDict):
    expected: int
    valid: int
    error: int
    cache_miss: int
    created_at: str


def get_cache_stats() -> CacheStats:
    statuses = [get_dataset_cache_status(dataset_name) for dataset_name in get_refreshed_hf_dataset_names()]
    return {
        "expected": len(statuses),
        "valid": len([status["dataset_name"] for status in statuses if status["status"] == "valid"]),
        "error": len([status["dataset_name"] for status in statuses if status["status"] == "error"]),
        "cache_miss": len([status["dataset_name"] for status in statuses if status["status"] == "cache_miss"]),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
