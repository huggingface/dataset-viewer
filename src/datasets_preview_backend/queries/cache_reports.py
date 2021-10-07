from typing import Dict, List, TypedDict, Union

from datasets_preview_backend.cache import cache, memoize  # type: ignore
from datasets_preview_backend.cache_entries import CacheEntry, get_cache_entries
from datasets_preview_backend.config import CACHE_SHORT_TTL_SECONDS
from datasets_preview_backend.exceptions import StatusError
from datasets_preview_backend.types import StatusErrorContent


class CacheReport(TypedDict):
    endpoint: str
    kwargs: Dict[str, Union[str, int]]
    status: str
    error: Union[StatusErrorContent, None]


class CacheReports(TypedDict):
    reports: List[CacheReport]


def entry_to_report(entry: CacheEntry) -> CacheReport:
    return {
        "endpoint": entry["endpoint"],
        "kwargs": entry["kwargs"],
        "status": entry["status"],
        "error": entry["error"].as_content() if isinstance(entry["error"], StatusError) else None,
    }


@memoize(cache=cache, expire=CACHE_SHORT_TTL_SECONDS)  # type:ignore
def get_cache_reports() -> CacheReports:
    return {"reports": [entry_to_report(entry) for entry in get_cache_entries()]}
