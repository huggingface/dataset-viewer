import logging
import time

from libqueue.queue import get_dataset_dump_by_status, get_split_dump_by_status
from starlette.requests import Request
from starlette.responses import Response

from api.config import MAX_AGE_SHORT_SECONDS
from api.routes._utils import get_response

logger = logging.getLogger(__name__)


async def queue_dump_waiting_started_endpoint(_: Request) -> Response:
    logger.info("/queue-dump-waiting-started")
    return get_response(
        {
            "datasets": get_dataset_dump_by_status(waiting_started=True),
            "splits": get_split_dump_by_status(waiting_started=True),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        200,
        MAX_AGE_SHORT_SECONDS,
    )
