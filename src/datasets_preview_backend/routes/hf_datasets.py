import logging

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.config import MAX_AGE_LONG_SECONDS
from datasets_preview_backend.io.cache import get_datasets_count_by_cache_status
from datasets_preview_backend.models.hf_dataset import get_hf_datasets
from datasets_preview_backend.routes._utils import get_response

logger = logging.getLogger(__name__)


async def hf_datasets_endpoint(_: Request) -> Response:
    logger.info("/hf-datasets")
    content = {"datasets": get_hf_datasets()}
    return get_response(content, 200, MAX_AGE_LONG_SECONDS)


def is_community(dataset_name: str) -> bool:
    return "/" in dataset_name


async def hf_datasets_count_by_cache_status_endpoint(_: Request) -> Response:
    logger.info("/hf-datasets-count-by-cache-status")
    dataset_names = get_hf_datasets()
    canonical = [x["id"] for x in dataset_names if not is_community(x["id"])]
    community = [x["id"] for x in dataset_names if is_community(x["id"])]
    content = {
        "canonical": get_datasets_count_by_cache_status(canonical),
        "community": get_datasets_count_by_cache_status(community),
    }
    return get_response(content, 200, MAX_AGE_LONG_SECONDS)
