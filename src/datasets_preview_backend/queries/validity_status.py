import time
from typing import List, TypedDict, cast

from datasets_preview_backend.cache_entries import CacheEntry, get_cache_entries
from datasets_preview_backend.queries.datasets import get_datasets
from datasets_preview_backend.types import ConfigsContent, SplitsContent


class DatasetStatus(TypedDict):
    dataset: str
    status: str


class DatasetsStatus(TypedDict):
    datasets: List[DatasetStatus]


class DatasetsByStatus(TypedDict):
    valid: List[str]
    error: List[str]
    cache_expired: List[str]
    cache_miss: List[str]
    created_at: str


class ExpectedEntries(TypedDict):
    expected_entries: List[CacheEntry]
    missing: int


def get_dataset_expected_entries(*, entries: List[CacheEntry], dataset: str) -> ExpectedEntries:
    dataset_entries = [
        entry for entry in entries if ("dataset" in entry["kwargs"] and entry["kwargs"]["dataset"] == dataset)
    ]

    expected_entries: List[CacheEntry] = []
    missing: int = 0

    configs_entries = [entry for entry in dataset_entries if entry["endpoint"] == "/configs"]
    if len(configs_entries) > 1:
        raise Exception("a dataset should have at most one /configs entry")
    elif len(configs_entries) == 0:
        missing += 1
    else:
        expected_entries.append(configs_entries[0])

        configs_content = cast(ConfigsContent, configs_entries[0]["content"])
        try:
            configItems = configs_content["configs"]
        except TypeError:
            configItems = []
        for config in [configItem["config"] for configItem in configItems]:
            infos_entries = [
                entry
                for entry in dataset_entries
                if entry["endpoint"] == "/infos" and entry["kwargs"]["config"] == config
            ]
            if len(infos_entries) > 1:
                raise Exception("a (dataset,config) tuple should have at most one /infos entry")
            elif len(infos_entries) == 0:
                missing += 1
            else:
                expected_entries.append(infos_entries[0])

            splits_entries = [
                entry
                for entry in dataset_entries
                if entry["endpoint"] == "/splits" and entry["kwargs"]["config"] == config
            ]
            if len(splits_entries) > 1:
                raise Exception("a (dataset,config) tuple should have at most one /splits entry")
            elif len(splits_entries) == 0:
                missing += 1
            else:
                expected_entries.append(splits_entries[0])

                splits_content = cast(SplitsContent, splits_entries[0]["content"])
                try:
                    splitItems = splits_content["splits"]
                except TypeError:
                    splitItems = []
                for split in [splitItem["split"] for splitItem in splitItems]:
                    rows_entries = [
                        entry
                        for entry in dataset_entries
                        if entry["endpoint"] == "/rows"
                        and entry["kwargs"]["config"] == config
                        and entry["kwargs"]["split"] == split
                    ]
                    if len(rows_entries) > 1:
                        raise Exception("a (dataset,config,split) tuple should have at most one /rows entry")
                    elif len(splits_entries) == 0:
                        missing += 1
                    else:
                        expected_entries.append(rows_entries[0])

    return {"expected_entries": expected_entries, "missing": missing}


def get_dataset_status(*, entries: List[CacheEntry], dataset: str) -> str:
    expected_entries = get_dataset_expected_entries(entries=entries, dataset=dataset)
    if any(r["status"] == "error" for r in expected_entries["expected_entries"]):
        return "error"
    elif (
        any(r["status"] == "cache_miss" for r in expected_entries["expected_entries"])
        or expected_entries["missing"] > 0
    ):
        return "cache_miss"
    elif any(r["status"] == "cache_expired" for r in expected_entries["expected_entries"]):
        return "cache_expired"
    return "valid"


def get_validity_status() -> DatasetsStatus:
    dataset_content = get_datasets()
    datasets = [datasetItem["dataset"] for datasetItem in dataset_content["datasets"]]
    entries = get_cache_entries()
    return {
        "datasets": [
            {"dataset": dataset, "status": get_dataset_status(entries=entries, dataset=dataset)}
            for dataset in datasets
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
        "cache_expired": [
            dataset_status["dataset"]
            for dataset_status in status["datasets"]
            if dataset_status["status"] == "cache_expired"
        ],
        "cache_miss": [
            dataset_status["dataset"]
            for dataset_status in status["datasets"]
            if dataset_status["status"] == "cache_miss"
        ],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
