import time
from typing import List, TypedDict, Union

from datasets_preview_backend.exceptions import StatusErrorContent
from datasets_preview_backend.models.dataset import get_dataset_cache_status
from datasets_preview_backend.models.hf_dataset import (
    HFDataset,
    get_refreshed_hf_datasets,
)


class CacheReport(TypedDict):
    dataset: str
    tags: List[str]
    downloads: Union[int, None]
    status: str
    error: Union[StatusErrorContent, None]


class CacheReports(TypedDict):
    reports: List[CacheReport]
    created_at: str


# we remove the content because it's too heavy
def get_dataset_report(hf_dataset: HFDataset) -> CacheReport:
    status = get_dataset_cache_status(hf_dataset["id"])
    return {
        "dataset": hf_dataset["id"],
        "tags": hf_dataset["tags"],
        "downloads": hf_dataset["downloads"],
        "status": status["status"],
        "error": status["error"],
    }


def get_cache_reports() -> CacheReports:
    return {
        "reports": [get_dataset_report(hf_dataset) for hf_dataset in get_refreshed_hf_datasets()],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
