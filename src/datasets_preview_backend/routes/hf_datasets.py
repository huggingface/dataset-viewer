import logging

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.config import MAX_AGE_LONG_SECONDS
from datasets_preview_backend.models.hf_dataset import get_hf_datasets
from datasets_preview_backend.routes._utils import get_response

logger = logging.getLogger(__name__)


async def hf_datasets_endpoint(_: Request) -> Response:
    logger.info("/datasets")
    content = {"datasets": get_hf_datasets()}
    return get_response(content, 200, MAX_AGE_LONG_SECONDS)
