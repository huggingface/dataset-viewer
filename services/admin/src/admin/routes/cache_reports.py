import logging
from typing import Optional

from libcache.simple_cache import (
    InvalidCursor,
    InvalidLimit,
    get_cache_reports_first_rows,
    get_cache_reports_splits,
)
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check
from admin.config import CACHE_REPORTS_NUM_RESULTS
from admin.utils import (
    AdminCustomError,
    Endpoint,
    InvalidParameterError,
    UnexpectedError,
    get_json_admin_error_response,
    get_json_ok_response,
)

logger = logging.getLogger(__name__)


def create_cache_reports_first_rows_endpoint(
    external_auth_url: Optional[str] = None, organization: Optional[str] = None
) -> Endpoint:
    async def cache_reports_first_rows_endpoint(request: Request) -> Response:
        try:
            cursor = request.query_params.get("cursor") or ""
            logger.info(f"/cache-reports/first-rows, cursor={cursor}")
            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(external_auth_url=external_auth_url, request=request, organization=organization)
            try:
                return get_json_ok_response(get_cache_reports_first_rows(cursor, CACHE_REPORTS_NUM_RESULTS))
            except InvalidCursor as e:
                raise InvalidParameterError("Invalid cursor.") from e
            except InvalidLimit as e:
                raise UnexpectedError(
                    "Invalid limit. CACHE_REPORTS_NUM_RESULTS must be a strictly positive integer."
                ) from e
        except AdminCustomError as e:
            return get_json_admin_error_response(e)
        except Exception:
            return get_json_admin_error_response(UnexpectedError("Unexpected error."))

    return cache_reports_first_rows_endpoint


def create_cache_reports_splits_endpoint(
    external_auth_url: Optional[str] = None, organization: Optional[str] = None
) -> Endpoint:
    async def cache_reports_splits_endpoint(request: Request) -> Response:
        try:
            cursor = request.query_params.get("cursor") or ""
            logger.info(f"/cache-reports/splits, cursor={cursor}")
            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(external_auth_url=external_auth_url, request=request, organization=organization)
            try:
                return get_json_ok_response(get_cache_reports_splits(cursor, CACHE_REPORTS_NUM_RESULTS))
            except InvalidCursor as e:
                raise InvalidParameterError("Invalid cursor.") from e
            except InvalidLimit as e:
                raise UnexpectedError(
                    "Invalid limit. CACHE_REPORTS_NUM_RESULTS must be a strictly positive integer."
                ) from e
        except AdminCustomError as e:
            return get_json_admin_error_response(e)
        except Exception:
            return get_json_admin_error_response(UnexpectedError("Unexpected error."))

    return cache_reports_splits_endpoint
