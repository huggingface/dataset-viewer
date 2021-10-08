import itertools
import time
from typing import List, TypedDict

from datasets_preview_backend.cache_entries import CacheEntry, get_expected_entries


class DatasetStatus(TypedDict):
    dataset: str
    status: str


class DatasetsStatus(TypedDict):
    datasets: List[DatasetStatus]


class DatasetsByStatus(TypedDict):
    valid: List[str]
    error: List[str]
    cache_miss: List[str]
    created_at: str


def get_entries_status(entries: List[CacheEntry]) -> str:
    if any(r["status"] == "error" for r in entries):
        return "error"
    elif any(r["status"] == "cache_miss" for r in entries):
        return "cache_miss"
    elif all(r["status"] == "valid" for r in entries):
        return "valid"
    raise Exception("should not occur")


def get_entry_dataset(cache_entry: CacheEntry) -> str:
    return cache_entry["kwargs"]["dataset"]


def get_validity_status() -> DatasetsStatus:
    entries = sorted(get_expected_entries(), key=get_entry_dataset)
    # print([a for a in sorted(itertools.groupby(entries, get_entry_dataset)]))
    return {
        "datasets": [
            {"dataset": dataset, "status": get_entries_status(list(dataset_entries))}
            for dataset, dataset_entries in itertools.groupby(entries, get_entry_dataset)
        ]
    }


def get_valid_datasets() -> DatasetsByStatus:
    status = get_validity_status()
    return {
        "valid": [
            dataset_status["dataset"] for dataset_status in status["datasets"] if dataset_status["status"] == "valid"
        ],
        "error": [
            dataset_status["dataset"] for dataset_status in status["datasets"] if dataset_status["status"] == "error"
        ],
        "cache_miss": [
            dataset_status["dataset"]
            for dataset_status in status["datasets"]
            if dataset_status["status"] == "cache_miss"
        ],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
