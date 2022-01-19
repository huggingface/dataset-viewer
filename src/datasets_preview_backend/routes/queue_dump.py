import logging

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.config import MAX_AGE_SHORT_SECONDS
from datasets_preview_backend.io.queue import (
    get_dataset_dump_by_status,
    get_split_dump_by_status,
)
from datasets_preview_backend.routes._utils import get_response

logger = logging.getLogger(__name__)


async def queue_dump_endpoint(_: Request) -> Response:
    logger.info("/queue-dump")
    return get_response(
        {"datasets": get_dataset_dump_by_status(), "splits": get_split_dump_by_status()}, 200, MAX_AGE_SHORT_SECONDS
    )
