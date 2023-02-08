# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.processing_graph import ProcessingStep
from libcommon.simple_cache import (
    InvalidCursor,
    InvalidLimit,
    get_cache_reports_with_content,
)
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check
from admin.utils import (
    AdminCustomError,
    Endpoint,
    InvalidParameterError,
    UnexpectedError,
    get_json_admin_error_response,
    get_json_ok_response,
)


def create_cache_reports_with_content_endpoint(
    processing_step: ProcessingStep,
    cache_reports_with_content_num_results: int,
    max_age: int,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
) -> Endpoint:
    async def cache_reports_with_content_endpoint(request: Request) -> Response:
        try:
            cursor = request.query_params.get("cursor") or ""
            logging.info(f"Cache reports with content for {processing_step.cache_kind}, cursor={cursor}")
            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(external_auth_url=external_auth_url, request=request, organization=organization)
            try:
                return get_json_ok_response(
                    get_cache_reports_with_content(
                        kind=processing_step.cache_kind, cursor=cursor, limit=cache_reports_with_content_num_results
                    ),
                    max_age=max_age,
                )
            except InvalidCursor as e:
                raise InvalidParameterError("Invalid cursor.") from e
            except InvalidLimit as e:
                raise UnexpectedError(
                    "Invalid limit. CACHE_REPORTS_WITH_CONTENT_NUM_RESULTS must be a strictly positive integer."
                ) from e
        except AdminCustomError as e:
            return get_json_admin_error_response(e, max_age=max_age)
        except Exception as e:
            return get_json_admin_error_response(UnexpectedError("Unexpected error.", e), max_age=max_age)

    return cache_reports_with_content_endpoint
