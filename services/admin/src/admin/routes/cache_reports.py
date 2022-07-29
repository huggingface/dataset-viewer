import logging

from libcache.simple_cache import (
    InvalidCursor,
    InvalidLimit,
    get_cache_reports_first_rows,
    get_cache_reports_splits_next,
)
from starlette.requests import Request
from starlette.responses import Response

from admin.config import CACHE_REPORTS_NUM_RESULTS
from admin.utils import (
    AdminCustomError,
    InvalidParameterError,
    UnexpectedError,
    get_json_admin_error_response,
    get_json_ok_response,
)

logger = logging.getLogger(__name__)


async def cache_reports_first_rows_endpoint(request: Request) -> Response:
    try:
        cursor = request.query_params.get("cursor") or ""
        logger.info(f"/cache-reports/first-rows, cursor={cursor}")
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


async def cache_reports_splits_next_endpoint(request: Request) -> Response:
    try:
        cursor = request.query_params.get("cursor") or ""
        logger.info(f"/cache-reports/splits-next, cursor={cursor}")
        try:
            return get_json_ok_response(get_cache_reports_splits_next(cursor, CACHE_REPORTS_NUM_RESULTS))
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
