# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libapi.exceptions import ApiError, InvalidParameterError, UnexpectedApiError
from libapi.utils import Endpoint, get_json_api_error_response, get_json_ok_response
from libcommon.simple_cache import InvalidCursor, InvalidLimit, get_cache_reports
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check


def create_cache_reports_endpoint(
    cache_kind: str,
    cache_reports_num_results: int,
    max_age: int,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Endpoint:
    async def cache_reports_endpoint(request: Request) -> Response:
        try:
            cursor = request.query_params.get("cursor") or ""
            logging.info(f"Cache reports for {cache_kind}, cursor={cursor}")
            # if auth_check fails, it will raise an exception that will be caught below
            await auth_check(
                external_auth_url=external_auth_url,
                request=request,
                organization=organization,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            try:
                return get_json_ok_response(
                    get_cache_reports(
                        kind=cache_kind,
                        cursor=cursor,
                        limit=cache_reports_num_results,
                    ),
                    max_age=max_age,
                )
            except InvalidCursor as e:
                raise InvalidParameterError("Invalid cursor.") from e
            except InvalidLimit as e:
                raise UnexpectedApiError(
                    "Invalid limit. CACHE_REPORTS_NUM_RESULTS must be a strictly positive integer."
                ) from e
        except ApiError as e:
            return get_json_api_error_response(e, max_age=max_age)
        except Exception as e:
            return get_json_api_error_response(UnexpectedApiError("Unexpected error.", e), max_age=max_age)

    return cache_reports_endpoint
