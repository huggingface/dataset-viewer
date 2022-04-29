import logging
import time

from libqueue.queue import (
    get_dataset_jobs_count_by_status,
    get_split_jobs_count_by_status,
)
from starlette.requests import Request
from starlette.responses import Response

from api_service.config import MAX_AGE_SHORT_SECONDS
from api_service.routes._utils import get_response

logger = logging.getLogger(__name__)


async def queue_stats_endpoint(_: Request) -> Response:
    logger.info("/queue")
    content = {
        "datasets": get_dataset_jobs_count_by_status(),
        "splits": get_split_jobs_count_by_status(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return get_response(content, 200, MAX_AGE_SHORT_SECONDS)
