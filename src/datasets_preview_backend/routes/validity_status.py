import logging
import time
from typing import List, TypedDict

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.config import MAX_AGE_SHORT_SECONDS
from datasets_preview_backend.io.mongo import get_dataset_cache
from datasets_preview_backend.models.hf_dataset import get_hf_dataset_names
from datasets_preview_backend.routes._utils import get_response

logger = logging.getLogger(__name__)


class DatasetsByStatus(TypedDict):
    valid: List[str]
    error: List[str]
    cache_miss: List[str]
    created_at: str


# TODO: improve by "counting" mongo entries instead
def get_valid_datasets() -> DatasetsByStatus:
    # TODO: cache get_hf_datasets_names?
    dataset_caches = [get_dataset_cache(dataset_name) for dataset_name in get_hf_dataset_names()]
    return {
        "valid": [dataset_cache.dataset_name for dataset_cache in dataset_caches if dataset_cache.status == "valid"],
        "error": [dataset_cache.dataset_name for dataset_cache in dataset_caches if dataset_cache.status == "error"],
        "cache_miss": [
            dataset_cache.dataset_name for dataset_cache in dataset_caches if dataset_cache.status == "cache_miss"
        ],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


async def valid_datasets_endpoint(_: Request) -> Response:
    logger.info("/valid")
    return get_response(get_valid_datasets(), 200, MAX_AGE_SHORT_SECONDS)
