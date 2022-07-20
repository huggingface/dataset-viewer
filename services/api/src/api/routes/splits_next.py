import logging

from libcache.simple_cache import DoesNotExist, HTTPStatus, get_splits_response
from libqueue.queue import is_splits_response_in_process
from libutils.exceptions import Status400Error, Status500Error
from starlette.requests import Request
from starlette.responses import Response

from api.config import MAX_AGE_LONG_SECONDS, MAX_AGE_SHORT_SECONDS
from api.routes._utils import get_response

logger = logging.getLogger(__name__)


async def splits_endpoint_next(request: Request) -> Response:
    dataset_name = request.query_params.get("dataset")
    logger.info(f"/splits-next, dataset={dataset_name}")

    if not isinstance(dataset_name, str):
        return get_response(Status400Error("Parameter 'dataset' is required").as_content(), 400, MAX_AGE_SHORT_SECONDS)
    try:
        response, http_status = get_splits_response(dataset_name)
        return get_response(
            response,
            int(http_status.value),
            MAX_AGE_LONG_SECONDS if http_status == HTTPStatus.OK else MAX_AGE_SHORT_SECONDS,
        )
    except DoesNotExist:
        if is_splits_response_in_process(dataset_name):
            return get_response(
                Status500Error("The list of splits is not ready yet. Please retry later.").as_content(),
                500,
                MAX_AGE_SHORT_SECONDS,
            )
        else:
            return get_response(Status400Error("Not found").as_content(), 400, MAX_AGE_SHORT_SECONDS)
