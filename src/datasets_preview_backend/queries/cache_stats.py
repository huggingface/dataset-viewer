from typing import Dict, List, TypedDict

from datasets_preview_backend.responses import memoized_functions
from datasets_preview_backend.cache_reports import get_cache_reports, ArgsCacheStats


class EndpointCacheStats(TypedDict):
    endpoint: str
    expected: int
    cached: int
    expired: int
    error: int
    valid: int


class CacheStats(TypedDict):
    endpoints: Dict[str, EndpointCacheStats]


def get_endpoint_report(endpoint: str, args_reports: List[ArgsCacheStats]) -> EndpointCacheStats:
    expected = args_reports
    cached = [d for d in expected if d["is_cached"]]
    expired = [d for d in expected if d["is_expired"]]
    error = [d for d in expected if d["is_error"]]
    valid = [d for d in expected if d["is_valid"]]
    return {
        "endpoint": endpoint,
        "expected": len(expected),
        "cached": len(cached),
        "expired": len(expired),
        "error": len(error),
        "valid": len(valid),
    }


def get_cache_stats() -> CacheStats:
    reports = get_cache_reports()

    endpoints = {
        endpoint: get_endpoint_report(endpoint, [report for report in reports if report["endpoint"] == endpoint])
        for endpoint in memoized_functions
    }

    return {"endpoints": endpoints}
