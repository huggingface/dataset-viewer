import logging
import time
from typing import Optional

from libcache.cache import (
    get_valid_or_stale_dataset_names,
    is_dataset_name_valid_or_stale,
)
from starlette.requests import Request
from starlette.responses import Response

from api.authentication import auth_check
from api.utils import (
    ApiCustomError,
    Endpoint,
    MissingRequiredParameterError,
    UnexpectedError,
    are_valid_parameters,
    get_json_api_error_response,
    get_json_ok_response,
)

logger = logging.getLogger(__name__)


async def valid_datasets_endpoint(_: Request) -> Response:
    try:
        logger.info("/valid")
        content = {
            "valid": get_valid_or_stale_dataset_names(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return get_json_ok_response(content)
    except Exception:
        return get_json_api_error_response(UnexpectedError("Unexpected error."))


def create_is_valid_endpoint(external_auth_url: Optional[str] = None) -> Endpoint:
    async def is_valid_endpoint(request: Request) -> Response:
        try:
            dataset_name = request.query_params.get("dataset")
            logger.info(f"/is-valid, dataset={dataset_name}")
            if not are_valid_parameters([dataset_name]):
                raise MissingRequiredParameterError("Parameter 'dataset' is required")
            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(dataset_name, external_auth_url=external_auth_url, request=request)
            content = {
                "valid": is_dataset_name_valid_or_stale(dataset_name),
            }
            return get_json_ok_response(content)
        except ApiCustomError as e:
            return get_json_api_error_response(e)
        except Exception:
            return get_json_api_error_response(UnexpectedError("Unexpected error."))

    return is_valid_endpoint
