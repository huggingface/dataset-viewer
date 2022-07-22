import logging
import time

from libcache.simple_cache import (
    get_first_rows_response_reports,
    get_splits_response_reports,
)
from starlette.requests import Request
from starlette.responses import Response

from admin.config import MAX_AGE_SHORT_SECONDS
from admin.routes._utils import get_response

logger = logging.getLogger(__name__)


async def cache_reports_endpoint(_: Request) -> Response:
    logger.info("/cache-reports")
    content = {
        "/splits-next": get_splits_response_reports(),
        "/first-rows": get_first_rows_response_reports(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return get_response(content, 200, MAX_AGE_SHORT_SECONDS)
