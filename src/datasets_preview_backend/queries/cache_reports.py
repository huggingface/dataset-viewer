import time
from typing import List, TypedDict, Union

from datasets_preview_backend.dataset_entries import (
    get_dataset_cache_status,
    get_refreshed_dataset_names,
)
from datasets_preview_backend.exceptions import StatusErrorContent


class CacheReport(TypedDict):
    dataset: str
    status: str
    error: Union[StatusErrorContent, None]


class CacheReports(TypedDict):
    reports: List[CacheReport]
    created_at: str


# we remove the content because it's too heavy
def get_dataset_report(dataset: str) -> CacheReport:
    status = get_dataset_cache_status(dataset)
    return {
        "dataset": dataset,
        "status": status["status"],
        "error": status["error"],
    }


def get_cache_reports() -> CacheReports:
    return {
        "reports": [get_dataset_report(dataset) for dataset in get_refreshed_dataset_names()],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
