from typing import Dict, List, TypedDict

from datasets_preview_backend.cache_reports import ArgsCacheStats, get_cache_reports
from datasets_preview_backend.responses import memoized_functions


class EndpointCacheStats(TypedDict):
    endpoint: str
    expected: int
    valid: int
    error: int
    cache_expired: int
    cache_miss: int


class CacheStats(TypedDict):
    endpoints: Dict[str, EndpointCacheStats]


def get_endpoint_report(endpoint: str, args_reports: List[ArgsCacheStats]) -> EndpointCacheStats:
    return {
        "endpoint": endpoint,
        "expected": len(args_reports),
        "valid": len([d for d in args_reports if d["status"] == "valid"]),
        "error": len([d for d in args_reports if d["status"] == "error"]),
        "cache_expired": len([d for d in args_reports if d["status"] == "cache_expired"]),
        "cache_miss": len([d for d in args_reports if d["status"] == "cache_miss"]),
    }


def get_cache_stats() -> CacheStats:
    reports = get_cache_reports()

    endpoints = {
        endpoint: get_endpoint_report(endpoint, [report for report in reports if report["endpoint"] == endpoint])
        for endpoint in memoized_functions
    }

    return {"endpoints": endpoints}
