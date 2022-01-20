import logging
import time

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.config import MAX_AGE_SHORT_SECONDS
from datasets_preview_backend.io.cache import (
    get_datasets_count_by_status,
    get_splits_count_by_status,
)
from datasets_preview_backend.routes._utils import get_response

logger = logging.getLogger(__name__)


async def cache_stats_endpoint(_: Request) -> Response:
    logger.info("/cache")
    content = {
        "datasets": get_datasets_count_by_status(),
        "splits": get_splits_count_by_status(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return get_response(content, 200, MAX_AGE_SHORT_SECONDS)
