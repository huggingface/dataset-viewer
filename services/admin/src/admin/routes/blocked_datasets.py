# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libapi.exceptions import ApiError, UnexpectedApiError
from libapi.utils import Endpoint, get_json_api_error_response, get_json_ok_response
from libcommon.queue.dataset_blockages import get_blocked_datasets
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check


def create_blocked_datasets_endpoint(
    max_age: int,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Endpoint:
    async def blocked_datasets_endpoint(request: Request) -> Response:
        logging.info("/blocked-datasets")
        try:
            # if auth_check fails, it will raise an exception that will be caught below
            await auth_check(
                external_auth_url=external_auth_url,
                request=request,
                organization=organization,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            return get_json_ok_response(
                {"blocked_datasets": get_blocked_datasets()},
                max_age=max_age,
            )
        except ApiError as e:
            return get_json_api_error_response(e, max_age=max_age)
        except Exception as e:
            return get_json_api_error_response(UnexpectedApiError("Unexpected error.", e), max_age=max_age)

    return blocked_datasets_endpoint
