import time
from typing import Dict, List, TypedDict, Union

from datasets_preview_backend.cache_entries import CacheEntry, get_cache_entries
from datasets_preview_backend.exceptions import StatusError
from datasets_preview_backend.types import StatusErrorContent


class CacheReport(TypedDict):
    endpoint: str
    kwargs: Dict[str, Union[str, int]]
    status: str
    expire: Union[str, None]
    error: Union[StatusErrorContent, None]


class CacheReports(TypedDict):
    reports: List[CacheReport]
    created_at: str


def entry_to_report(entry: CacheEntry) -> CacheReport:
    return {
        "endpoint": entry["endpoint"],
        "kwargs": entry["kwargs"],
        "status": entry["status"],
        "expire": None
        if entry["expire_time"] is None
        else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(entry["expire_time"])),
        "error": entry["error"].as_content() if isinstance(entry["error"], StatusError) else None,
    }


def get_cache_reports() -> CacheReports:
    return {
        "reports": [entry_to_report(entry) for entry in get_cache_entries()],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
