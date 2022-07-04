import logging

from libcache.simple_cache import DoesNotExist, HTTPStatus, get_first_rows_response
from libutils.exceptions import Status400Error
from starlette.requests import Request
from starlette.responses import Response

from api.config import MAX_AGE_LONG_SECONDS, MAX_AGE_SHORT_SECONDS
from api.routes._utils import get_response

logger = logging.getLogger(__name__)


async def first_rows_endpoint(request: Request) -> Response:
    dataset_name = request.query_params.get("dataset")
    config_name = request.query_params.get("config")
    split_name = request.query_params.get("split")
    logger.info(f"/rows, dataset={dataset_name}, config={config_name}, split={split_name}")

    if not isinstance(dataset_name, str) or not isinstance(config_name, str) or not isinstance(split_name, str):
        return get_response(
            Status400Error("Parameters 'dataset', 'config' and 'split' are required").as_content(),
            400,
            MAX_AGE_SHORT_SECONDS,
        )
    try:
        response, http_status = get_first_rows_response(dataset_name, config_name, split_name)
        return get_response(
            response,
            int(http_status.value),
            MAX_AGE_LONG_SECONDS if http_status == HTTPStatus.OK else MAX_AGE_SHORT_SECONDS,
        )
    except DoesNotExist:
        return get_response(Status400Error("Not found").as_content(), 400, MAX_AGE_SHORT_SECONDS)
