import logging

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.config import MAX_AGE_LONG_SECONDS
from datasets_preview_backend.exceptions import StatusError
from datasets_preview_backend.io.cache import get_splits_response
from datasets_preview_backend.routes._utils import get_response

logger = logging.getLogger(__name__)


async def splits_endpoint(request: Request) -> Response:
    dataset_name = request.query_params.get("dataset")
    logger.info(f"/splits, dataset={dataset_name}")

    try:
        if not isinstance(dataset_name, str):
            raise StatusError("Parameter 'dataset' is required", 400)
        splits_response, splits_error, status_code = get_splits_response(dataset_name)
        return get_response(splits_response or splits_error, status_code, MAX_AGE_LONG_SECONDS)
    except StatusError as err:
        return get_response(err.as_content(), err.status_code, MAX_AGE_LONG_SECONDS)
