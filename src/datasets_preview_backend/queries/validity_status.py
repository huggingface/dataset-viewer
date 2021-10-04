from typing import List, TypedDict, cast

# from datasets_preview_backend.responses import memoized_functions
from datasets_preview_backend.cache_reports import ArgsCacheStats, get_cache_reports
from datasets_preview_backend.queries.datasets import get_datasets
from datasets_preview_backend.types import ConfigsContent, SplitsContent


class DatasetValidityStatus(TypedDict):
    dataset: str
    is_valid: bool
    # reports: List[ArgsCacheStats]


class ValidityStatus(TypedDict):
    datasets: List[DatasetValidityStatus]


class ValidDatasets(TypedDict):
    datasets: List[str]


def is_dataset_valid(*, reports: List[ArgsCacheStats], dataset: str) -> bool:
    # we could partition the list
    dataset_reports = [
        report for report in reports if ("dataset" in report["kwargs"] and report["kwargs"]["dataset"] == dataset)
    ]

    configs_reports = [report for report in dataset_reports if report["endpoint"] == "/configs"]
    if len(configs_reports) != 1 or not configs_reports[0]["is_valid"]:
        return False

    configs = [configItem["config"] for configItem in cast(ConfigsContent, configs_reports[0]["content"])["configs"]]
    for config in configs:
        infos_reports = [
            report
            for report in dataset_reports
            if report["endpoint"] == "/infos" and report["kwargs"]["config"] == config
        ]
        if len(infos_reports) != 1 or not infos_reports[0]["is_valid"]:
            return False

        splits_reports = [
            report
            for report in dataset_reports
            if report["endpoint"] == "/splits" and report["kwargs"]["config"] == config
        ]
        if len(splits_reports) != 1 or not splits_reports[0]["is_valid"]:
            return False

        splits = [splitItem["split"] for splitItem in cast(SplitsContent, splits_reports[0]["content"])["splits"]]
        for split in splits:
            rows_reports = [
                report
                for report in dataset_reports
                if report["endpoint"] == "/rows"
                and report["kwargs"]["config"] == config
                and report["kwargs"]["split"] == split
            ]
            if len(rows_reports) != 1 or not rows_reports[0]["is_valid"]:
                return False

    return True


def get_validity_status() -> ValidityStatus:
    dataset_content = get_datasets()
    datasets = [datasetItem["dataset"] for datasetItem in dataset_content["datasets"]]
    reports = get_cache_reports()
    return {
        "datasets": [
            {"dataset": dataset, "is_valid": is_dataset_valid(reports=reports, dataset=dataset)}
            for dataset in datasets
        ]
    }


def get_valid_datasets() -> ValidDatasets:
    # we only report the cached datasets as valid
    # as we rely on cache warming at startup (otherwise, the first call would take too long - various hours)
    # note that warming can be done by 1. calling /datasets, then 2. calling /rows?dataset={dataset}
    # for all the datasets
    status = get_validity_status()
    return {
        "datasets": [dataset_status["dataset"] for dataset_status in status["datasets"] if dataset_status["is_valid"]]
    }
