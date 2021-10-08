import time
from typing import Dict, List, TypedDict

from datasets_preview_backend.cache_entries import CacheEntry, get_expected_entries


class EndpointCacheStats(TypedDict):
    endpoint: str
    expected: int
    valid: int
    error: int
    cache_miss: int


class CacheStats(TypedDict):
    endpoints: Dict[str, EndpointCacheStats]
    created_at: str


def get_endpoint_report(endpoint: str, cache_entries: List[CacheEntry]) -> EndpointCacheStats:
    return {
        "endpoint": endpoint,
        "expected": len(cache_entries),
        "valid": len([d for d in cache_entries if d["status"] == "valid"]),
        "error": len([d for d in cache_entries if d["status"] == "error"]),
        "cache_miss": len([d for d in cache_entries if d["status"] == "cache_miss"]),
    }


def get_cache_stats() -> CacheStats:
    cache_entries = get_expected_entries()

    by_endpoint = {
        endpoint: get_endpoint_report(endpoint, [entry for entry in cache_entries if entry["endpoint"] == endpoint])
        for endpoint in [
            "/configs",
            "/infos",
            "/splits",
            "/rows",
        ]
    }

    return {"endpoints": by_endpoint, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
