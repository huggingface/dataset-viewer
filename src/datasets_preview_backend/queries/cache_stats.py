from typing import Dict, List, TypedDict

from datasets_preview_backend.cache import cache, memoize  # type: ignore
from datasets_preview_backend.cache_entries import (
    CacheEntry,
    get_cache_entries,
    memoized_functions,
)
from datasets_preview_backend.config import CACHE_SHORT_TTL_SECONDS


class EndpointCacheStats(TypedDict):
    endpoint: str
    expected: int
    valid: int
    error: int
    cache_expired: int
    cache_miss: int


class CacheStats(TypedDict):
    endpoints: Dict[str, EndpointCacheStats]


def get_endpoint_report(endpoint: str, cache_entries: List[CacheEntry]) -> EndpointCacheStats:
    return {
        "endpoint": endpoint,
        "expected": len(cache_entries),
        "valid": len([d for d in cache_entries if d["status"] == "valid"]),
        "error": len([d for d in cache_entries if d["status"] == "error"]),
        "cache_expired": len([d for d in cache_entries if d["status"] == "cache_expired"]),
        "cache_miss": len([d for d in cache_entries if d["status"] == "cache_miss"]),
    }


@memoize(cache=cache, expire=CACHE_SHORT_TTL_SECONDS)  # type:ignore
def get_cache_stats() -> CacheStats:
    cache_entries = get_cache_entries()

    endpoints = {
        endpoint: get_endpoint_report(endpoint, [entry for entry in cache_entries if entry["endpoint"] == endpoint])
        for endpoint in memoized_functions
    }

    return {"endpoints": endpoints}
