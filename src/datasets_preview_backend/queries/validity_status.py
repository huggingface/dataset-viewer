from typing import List, TypedDict, cast

from datasets_preview_backend.cache_reports import ArgsCacheStats, get_cache_reports
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


def get_dataset_status(*, reports: List[ArgsCacheStats], dataset: str) -> str:
    dataset_reports = [
        report for report in reports if ("dataset" in report["kwargs"] and report["kwargs"]["dataset"] == dataset)
    ]

    configs_reports = [report for report in dataset_reports if report["endpoint"] == "/configs"]
    if len(configs_reports) != 1:
        raise Exception("a dataset should have exactly one /configs report")
    configs_report = configs_reports[0]
    if configs_report["status"] != "valid":
        return configs_report["status"]

    configs = [configItem["config"] for configItem in cast(ConfigsContent, configs_report["content"])["configs"]]
    for config in configs:
        infos_reports = [
            report
            for report in dataset_reports
            if report["endpoint"] == "/infos" and report["kwargs"]["config"] == config
        ]
        if len(infos_reports) != 1:
            raise Exception("a (dataset,config) tuple should have exactly one /infos report")
        infos_report = infos_reports[0]
        if infos_report["status"] != "valid":
            return infos_report["status"]

        splits_reports = [
            report
            for report in dataset_reports
            if report["endpoint"] == "/splits" and report["kwargs"]["config"] == config
        ]
        if len(splits_reports) != 1:
            raise Exception("a (dataset,config) tuple should have exactly one /splits report")
        splits_report = splits_reports[0]
        if splits_report["status"] != "valid":
            return splits_report["status"]

        splits = [splitItem["split"] for splitItem in cast(SplitsContent, splits_report["content"])["splits"]]
        for split in splits:
            rows_reports = [
                report
                for report in dataset_reports
                if report["endpoint"] == "/rows"
                and report["kwargs"]["config"] == config
                and report["kwargs"]["split"] == split
            ]
            if len(rows_reports) != 1:
                raise Exception("a (dataset,config,split) tuple should have exactly one /rows report")
            rows_report = rows_reports[0]
            if rows_report["status"] != "valid":
                return rows_report["status"]

    return "valid"


def get_validity_status() -> DatasetsStatus:
    dataset_content = get_datasets()
    datasets = [datasetItem["dataset"] for datasetItem in dataset_content["datasets"]]
    reports = get_cache_reports()
    return {
        "datasets": [
            {"dataset": dataset, "status": get_dataset_status(reports=reports, dataset=dataset)}
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
    }
