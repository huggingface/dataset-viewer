# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Optional

from libcache.simple_cache import DoesNotExist, get_first_rows_response
from libqueue.queue import is_first_rows_response_in_process
from starlette.requests import Request
from starlette.responses import Response

from api.authentication import auth_check
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
            dataset_name = request.query_params.get("dataset")
            config_name = request.query_params.get("config")
            split_name = request.query_params.get("split")
            logger.info(f"/rows, dataset={dataset_name}, config={config_name}, split={split_name}")

            if not are_valid_parameters([dataset_name, config_name, split_name]):
                raise MissingRequiredParameterError("Parameters 'dataset', 'config' and 'split' are required")
            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(dataset_name, external_auth_url=external_auth_url, request=request)
            try:
                response, http_status, error_code = get_first_rows_response(dataset_name, config_name, split_name)
                if http_status == HTTPStatus.OK:
                    return get_json_ok_response(response)
                else:
                    return get_json_error_response(response, http_status, error_code)
            except DoesNotExist as e:
                if is_first_rows_response_in_process(dataset_name, config_name, split_name):
                    raise FirstRowsResponseNotReadyError(
                        "The list of the first rows is not ready yet. Please retry later."
                    ) from e
                else:
                    raise FirstRowsResponseNotFoundError("Not found.") from e
        except ApiCustomError as e:
            return get_json_api_error_response(e)
        except Exception as e:
            return get_json_api_error_response(UnexpectedError("Unexpected error.", e))

    return first_rows_endpoint
