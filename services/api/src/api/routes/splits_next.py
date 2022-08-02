import logging
from http import HTTPStatus

from libcache.simple_cache import DoesNotExist, get_splits_response
from libqueue.queue import is_splits_response_in_process
from starlette.requests import Request
from starlette.responses import Response

from api.authentication import auth_check
from api.utils import (
    ApiCustomError,
    Endpoint,
    MissingRequiredParameterError,
    SplitsResponseNotFoundError,
    SplitsResponseNotReadyError,
    UnexpectedError,
    are_valid_parameters,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
)

logger = logging.getLogger(__name__)


def create_splits_next_endpoint(external_auth_url: str = "") -> Endpoint:
    async def splits_next_endpoint(request: Request) -> Response:
        try:
            dataset_name = request.query_params.get("dataset")
            logger.info(f"/splits-next, dataset={dataset_name}")

            if not are_valid_parameters([dataset_name]):
                raise MissingRequiredParameterError("Parameter 'dataset' is required")
            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(dataset_name, external_auth_url=external_auth_url, request=request)
            try:
                response, http_status, error_code = get_splits_response(dataset_name)
                if http_status == HTTPStatus.OK:
                    return get_json_ok_response(response)
                else:
                    return get_json_error_response(response, http_status, error_code)
            except DoesNotExist as e:
                if is_splits_response_in_process(dataset_name):
                    raise SplitsResponseNotReadyError(
                        "The list of splits is not ready yet. Please retry later."
                    ) from e
                else:
                    raise SplitsResponseNotFoundError("Not found.") from e
        except ApiCustomError as e:
            return get_json_api_error_response(e)
        except Exception as err:
            return get_json_api_error_response(UnexpectedError("Unexpected error.", err))

    return splits_next_endpoint
