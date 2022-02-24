import logging

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.config import MAX_AGE_LONG_SECONDS, ROWS_MAX_BYTES
from datasets_preview_backend.exceptions import StatusError
from datasets_preview_backend.io.cache import get_rows_response
from datasets_preview_backend.routes._utils import get_response

logger = logging.getLogger(__name__)


async def rows_endpoint(request: Request) -> Response:
    dataset_name = request.query_params.get("dataset")
    config_name = request.query_params.get("config")
    split_name = request.query_params.get("split")
    logger.info(f"/rows, dataset={dataset_name}, config={config_name}, split={split_name}")

    try:
        if not isinstance(dataset_name, str) or not isinstance(config_name, str) or not isinstance(split_name, str):
            raise StatusError("Parameters 'dataset', 'config' and 'split' are required", 400)
        rows_response, rows_error, status_code = get_rows_response(
            dataset_name, config_name, split_name, ROWS_MAX_BYTES
        )
        return get_response(rows_response or rows_error, status_code, MAX_AGE_LONG_SECONDS)
    except StatusError as err:
        return get_response(err.as_content(), err.status_code, MAX_AGE_LONG_SECONDS)
