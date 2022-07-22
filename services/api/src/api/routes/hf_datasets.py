import logging

from huggingface_hub import list_datasets  # type: ignore
from libcache.cache import get_datasets_count_by_cache_status
from starlette.requests import Request
from starlette.responses import Response

from api.config import MAX_AGE_LONG_SECONDS
from api.routes._utils import get_response

logger = logging.getLogger(__name__)


def is_community(dataset_name: str) -> bool:
    return "/" in dataset_name


async def hf_datasets_count_by_cache_status_endpoint(_: Request) -> Response:
    logger.info("/hf-datasets-count-by-cache-status")
    # If an exception is raised, we let it propagate. Starlette will return a 500 error
    datasets = list_datasets(full=False)
    canonical = [x.id for x in datasets if not is_community(x.id)]
    community = [x.id for x in datasets if is_community(x.id)]
    content = {
        "canonical": get_datasets_count_by_cache_status(canonical),
        "community": get_datasets_count_by_cache_status(community),
    }
    return get_response(content, 200, MAX_AGE_LONG_SECONDS)
