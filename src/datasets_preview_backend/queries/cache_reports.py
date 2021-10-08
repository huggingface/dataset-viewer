import time
from typing import Dict, List, TypedDict, Union

from datasets_preview_backend.cache_entries import CacheEntry, get_expected_entries
from datasets_preview_backend.exceptions import StatusErrorContent


class CacheReport(TypedDict):
    endpoint: str
    kwargs: Dict[str, str]
    status: str
    error: Union[StatusErrorContent, None]


class CacheReports(TypedDict):
    reports: List[CacheReport]
    created_at: str


# we remove the content because it's too heavy
def entry_to_report(entry: CacheEntry) -> CacheReport:
    return {
        "endpoint": entry["endpoint"],
        "kwargs": entry["kwargs"],
        "status": entry["status"],
        "error": entry["error"],
    }


def get_cache_reports() -> CacheReports:
    return {
        "reports": [entry_to_report(entry) for entry in get_expected_entries()],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
