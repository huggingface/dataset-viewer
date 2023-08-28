# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libapi.exceptions import UnexpectedApiError
from libapi.utils import Endpoint, get_json_api_error_response, get_json_ok_response
from libcommon.prometheus import StepProfiler
from libcommon.simple_cache import get_contents_page
from starlette.requests import Request
from starlette.responses import Response


def create_hub_cache_endpoint(
    cache_kind: str,
    num_results_per_page: int,
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Endpoint:
    # this endpoint is used by the frontend to know which datasets support the dataset viewer
    # and to give their size (number of rows)
    async def hub_cache_endpoint(request: Request) -> Response:
        with StepProfiler(method="hub_cache_endpoint", step="all"):
            logging.info("/hub-cache")
            try:
                cursor = request.query_params.get("cursor")
                endpoint_url = str(request.url)
                # TODO: add auth to only allow the Hub to call this endpoint?
                result = get_contents_page(kind=cache_kind, limit=num_results_per_page, cursor=cursor)
                link = f"""<{endpoint_url}?cursor={result["cursor"]}>;rel="next" """ if result["cursor"] else ""
                return get_json_ok_response(result["contents"], max_age=max_age_long, headers={"Link": link})
            except Exception as e:
                return get_json_api_error_response(UnexpectedApiError("Unexpected error.", e), max_age=max_age_short)

    return hub_cache_endpoint
