import time
from typing import Dict, List, TypedDict

from datasets_preview_backend.cache import cache, memoize  # type: ignore
from datasets_preview_backend.cache_entries import (
    CacheEntry,
    get_cache_entries,
    memoized_functions,
)
from datasets_preview_backend.config import CACHE_SHORT_TTL_SECONDS


class ExpireWithin(TypedDict):
    name: str
    seconds: int
    number: int


class EndpointCacheStats(TypedDict):
    endpoint: str
    expected: int
    valid: int
    error: int
    cache_expired: int
    cache_miss: int
    expire_within: List[ExpireWithin]


class CacheStats(TypedDict):
    endpoints: Dict[str, EndpointCacheStats]
    created_at: str


def get_number_expire_within(cache_entries: List[CacheEntry], current_time: float, within: int) -> int:
    return len(
        [
            d
            for d in cache_entries
            if d["expire_time"] is not None
            and d["expire_time"] >= current_time
            and d["expire_time"] < current_time + within
        ]
    )


EXPIRE_THRESHOLDS = [
    ("1m", 60),
    ("10m", 10 * 60),
    ("1h", 60 * 60),
    ("10h", 10 * 60 * 60),
]


def get_endpoint_report(endpoint: str, cache_entries: List[CacheEntry], current_time: float) -> EndpointCacheStats:
    return {
        "endpoint": endpoint,
        "expected": len(cache_entries),
        "valid": len([d for d in cache_entries if d["status"] == "valid"]),
        "error": len([d for d in cache_entries if d["status"] == "error"]),
        "cache_expired": len([d for d in cache_entries if d["status"] == "cache_expired"]),
        "cache_miss": len([d for d in cache_entries if d["status"] == "cache_miss"]),
        "expire_within": [
            {
                "name": name,
                "seconds": seconds,
                "number": get_number_expire_within(cache_entries, current_time, seconds),
            }
            for (name, seconds) in EXPIRE_THRESHOLDS
        ],
    }


@memoize(cache=cache, expire=CACHE_SHORT_TTL_SECONDS)  # type:ignore
def get_cache_stats() -> CacheStats:
    cache_entries = get_cache_entries()
    current_time = time.time()

    endpoints = {
        endpoint: get_endpoint_report(
            endpoint, [entry for entry in cache_entries if entry["endpoint"] == endpoint], current_time
        )
        for endpoint in memoized_functions
    }

    return {"endpoints": endpoints, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_time))}
