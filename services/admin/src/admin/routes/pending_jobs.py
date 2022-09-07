import logging
import time

from libqueue.queue import get_first_rows_dump_by_status, get_splits_dump_by_status
from starlette.requests import Request
from starlette.responses import Response

from admin.config import MAX_AGE_SHORT_SECONDS
from admin.utils import get_response

logger = logging.getLogger(__name__)


async def pending_jobs_endpoint(_: Request) -> Response:
    logger.info("/pending-jobs")
    return get_response(
        {
            "/splits": get_splits_dump_by_status(waiting_started=True),
            "/first-rows": get_first_rows_dump_by_status(waiting_started=True),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        200,
        MAX_AGE_SHORT_SECONDS,
    )
