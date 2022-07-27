import logging
from http import HTTPStatus

from libcache.simple_cache import DoesNotExist, get_splits_response
from libqueue.queue import is_splits_response_in_process
from starlette.requests import Request
from starlette.responses import Response

from api.utils import (
    ApiCustomError,
    MissingRequiredParameterError,
    SplitsResponseNotFoundError,
    SplitsResponseNotReadyError,
    UnexpectedError,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
)

logger = logging.getLogger(__name__)


async def splits_endpoint_next(request: Request) -> Response:
    try:
        dataset_name = request.query_params.get("dataset")
        logger.info(f"/splits-next, dataset={dataset_name}")

        if not isinstance(dataset_name, str):
            raise MissingRequiredParameterError("Parameter 'dataset' is required")
        try:
            response, http_status, error_code = get_splits_response(dataset_name)
            if http_status == HTTPStatus.OK:
                return get_json_ok_response(response)
            else:
                return get_json_error_response(response, http_status, error_code)
        except DoesNotExist as e:
            if is_splits_response_in_process(dataset_name):
                raise SplitsResponseNotReadyError("The list of splits is not ready yet. Please retry later.") from e
            else:
                raise SplitsResponseNotFoundError("Not found.") from e
    except ApiCustomError as e:
        return get_json_api_error_response(e)
    except Exception:
        return get_json_api_error_response(UnexpectedError("Unexpected error."))
