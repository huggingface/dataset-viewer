import logging
import time

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.config import MAX_AGE_SHORT_SECONDS
from datasets_preview_backend.io.cache import get_datasets_reports
from datasets_preview_backend.routes._utils import get_response

logger = logging.getLogger(__name__)


async def cache_reports_endpoint(_: Request) -> Response:
    logger.info("/cache-reports")
    content = {
        "reports": get_datasets_reports(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return get_response(content, 200, MAX_AGE_SHORT_SECONDS)
