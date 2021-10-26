import logging
import time
from typing import TypedDict

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.config import MAX_AGE_SHORT_SECONDS
from datasets_preview_backend.io.mongo import get_dataset_cache
from datasets_preview_backend.models.hf_dataset import get_hf_dataset_names
from datasets_preview_backend.routes._utils import get_response

logger = logging.getLogger(__name__)


class CacheStats(TypedDict):
    expected: int
    valid: int
    error: int
    cache_miss: int
    created_at: str


# TODO: improve by "counting" mongo entries instead
def get_cache_stats() -> CacheStats:
    # TODO: cache get_hf_datasets_names?
    dataset_caches = [get_dataset_cache(dataset_name) for dataset_name in get_hf_dataset_names()]
    return {
        "expected": len(dataset_caches),
        "valid": len(
            [dataset_cache.dataset_name for dataset_cache in dataset_caches if dataset_cache.status == "valid"]
        ),
        "error": len(
            [dataset_cache.dataset_name for dataset_cache in dataset_caches if dataset_cache.status == "error"]
        ),
        "cache_miss": len(
            [dataset_cache.dataset_name for dataset_cache in dataset_caches if dataset_cache.status == "cache_miss"]
        ),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


async def cache_stats_endpoint(_: Request) -> Response:
    logger.info("/cache-reports")
    return get_response(get_cache_stats(), 200, MAX_AGE_SHORT_SECONDS)
