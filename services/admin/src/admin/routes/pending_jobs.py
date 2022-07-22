import logging
import time

from libqueue.queue import (
    get_dataset_dump_by_status,
    get_first_rows_dump_by_status,
    get_split_dump_by_status,
    get_splits_dump_by_status,
)
from starlette.requests import Request
from starlette.responses import Response

from admin.config import MAX_AGE_SHORT_SECONDS
from admin.routes._utils import get_response

logger = logging.getLogger(__name__)


async def pending_jobs_endpoint(_: Request) -> Response:
    logger.info("/pending-jobs")
    return get_response(
        {
            "/splits": get_dataset_dump_by_status(waiting_started=True),
            "/rows": get_split_dump_by_status(waiting_started=True),
            "/splits-next": get_splits_dump_by_status(waiting_started=True),
            "/first-rows": get_first_rows_dump_by_status(waiting_started=True),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        200,
        MAX_AGE_SHORT_SECONDS,
    )
