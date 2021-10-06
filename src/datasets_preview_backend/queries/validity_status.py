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


class ExpectedReports(TypedDict):
    expected_reports: List[ArgsCacheStats]
    missing: int


def get_dataset_expected_reports(*, reports: List[ArgsCacheStats], dataset: str) -> ExpectedReports:
    dataset_reports = [
        report for report in reports if ("dataset" in report["kwargs"] and report["kwargs"]["dataset"] == dataset)
    ]

    expected_reports: List[ArgsCacheStats] = []
    missing: int = 0

    configs_reports = [report for report in dataset_reports if report["endpoint"] == "/configs"]
    if len(configs_reports) > 1:
        raise Exception("a dataset should have at most one /configs report")
    elif len(configs_reports) == 0:
        missing += 1
    else:
        expected_reports.append(configs_reports[0])

        configs_content = cast(ConfigsContent, configs_reports[0]["content"])
        if configs_content is not None:
            for config in [configItem["config"] for configItem in configs_content["configs"]]:
                infos_reports = [
                    report
                    for report in dataset_reports
                    if report["endpoint"] == "/infos" and report["kwargs"]["config"] == config
                ]
                if len(infos_reports) > 1:
                    raise Exception("a (dataset,config) tuple should have at most one /infos report")
                elif len(infos_reports) == 0:
                    missing += 1
                else:
                    expected_reports.append(infos_reports[0])

                splits_reports = [
                    report
                    for report in dataset_reports
                    if report["endpoint"] == "/splits" and report["kwargs"]["config"] == config
                ]
                if len(splits_reports) > 1:
                    raise Exception("a (dataset,config) tuple should have at most one /splits report")
                elif len(splits_reports) == 0:
                    missing += 1
                else:
                    expected_reports.append(splits_reports[0])

                    splits_content = cast(SplitsContent, splits_reports[0]["content"])
                    if splits_content is not None:
                        for split in [splitItem["split"] for splitItem in splits_content["splits"]]:
                            rows_reports = [
                                report
                                for report in dataset_reports
                                if report["endpoint"] == "/rows"
                                and report["kwargs"]["config"] == config
                                and report["kwargs"]["split"] == split
                            ]
                            if len(rows_reports) > 1:
                                raise Exception("a (dataset,config,split) tuple should have at most one /rows report")
                            elif len(splits_reports) == 0:
                                missing += 1
                            else:
                                expected_reports.append(rows_reports[0])

    return {"expected_reports": expected_reports, "missing": missing}


def get_dataset_status(*, reports: List[ArgsCacheStats], dataset: str) -> str:
    expected_reports = get_dataset_expected_reports(reports=reports, dataset=dataset)
    if any(r["status"] == "error" for r in expected_reports["expected_reports"]):
        return "error"
    elif (
        any(r["status"] == "cache_miss" for r in expected_reports["expected_reports"])
        or expected_reports["missing"] > 0
    ):
        return "cache_miss"
    elif any(r["status"] == "cache_expired" for r in expected_reports["expected_reports"]):
        return "cache_expired"
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
