# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Optional

from libcache.simple_cache import DoesNotExist, get_first_rows_response
from starlette.requests import Request
from starlette.responses import Response

from api.authentication import auth_check
from api.dataset import is_first_rows_in_process
from api.utils import (
    ApiCustomError,
    Endpoint,
    FirstRowsResponseNotFoundError,
    FirstRowsResponseNotReadyError,
    MissingRequiredParameterError,
    UnexpectedError,
    are_valid_parameters,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
)

logger = logging.getLogger(__name__)


def create_first_rows_endpoint(
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    external_auth_url: Optional[str] = None,
) -> Endpoint:
    async def first_rows_endpoint(request: Request) -> Response:
        try:
            dataset = request.query_params.get("dataset")
            config = request.query_params.get("config")
            split = request.query_params.get("split")
            logger.info(f"/rows, dataset={dataset}, config={config}, split={split}")

            if not are_valid_parameters([dataset, config, split]):
                raise MissingRequiredParameterError("Parameters 'dataset', 'config' and 'split' are required")
            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(dataset, external_auth_url=external_auth_url, request=request)
            try:
                response, http_status, error_code = get_first_rows_response(dataset, config, split)
                if http_status == HTTPStatus.OK:
                    return get_json_ok_response(response)
                else:
                    return get_json_error_response(response, http_status, error_code)
            except DoesNotExist as e:
                # maybe the first-rows response is in process
                if is_first_rows_in_process(
                    dataset=dataset, config=config, split=split, hf_endpoint=hf_endpoint, hf_token=hf_token
                ):
                    raise FirstRowsResponseNotReadyError(
                        "The list of the first rows is not ready yet. Please retry later."
                    ) from e
                raise FirstRowsResponseNotFoundError("Not found.") from e
        except ApiCustomError as e:
            return get_json_api_error_response(e)
        except Exception as e:
            return get_json_api_error_response(UnexpectedError("Unexpected error.", e))

    return first_rows_endpoint
