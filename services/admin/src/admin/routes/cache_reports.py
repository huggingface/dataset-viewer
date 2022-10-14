# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Callable, Dict, Literal, Optional

from libcache.simple_cache import (
    InvalidCursor,
    InvalidLimit,
    get_cache_reports_features,
    get_cache_reports_first_rows,
    get_cache_reports_splits,
)
from starlette.requests import Request
from starlette.responses import Response

from ..authentication import auth_check
from ..config import CACHE_REPORTS_NUM_RESULTS
from ..utils import (
    AdminCustomError,
    Endpoint,
    InvalidParameterError,
    UnexpectedError,
    get_json_admin_error_response,
    get_json_ok_response,
)

logger = logging.getLogger(__name__)


EndpointName = Literal["features", "first-rows", "splits"]


get_cache_reports: Dict[EndpointName, Callable] = {
    "features": get_cache_reports_features,
    "first-rows": get_cache_reports_first_rows,
    "splits": get_cache_reports_splits,
}


def create_cache_reports_endpoint(
    endpoint: EndpointName, external_auth_url: Optional[str] = None, organization: Optional[str] = None
) -> Endpoint:
    get_cache_reports = get_cache_reports_features if endpoint == "features" else get_cache_reports_first_rows

    async def cache_reports_endpoint(request: Request) -> Response:
        try:
            cursor = request.query_params.get("cursor") or ""
            logger.info(f"/cache-reports/{endpoint}, cursor={cursor}")
            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(external_auth_url=external_auth_url, request=request, organization=organization)
            try:
                return get_json_ok_response(get_cache_reports(cursor, CACHE_REPORTS_NUM_RESULTS))
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

    return cache_reports_endpoint
