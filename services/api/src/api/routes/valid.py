# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libcache.simple_cache import get_valid_dataset_names, is_dataset_name_valid
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


async def valid_endpoint(_: Request) -> Response:
    try:
        logging.info("/valid")
        content = {"valid": get_valid_dataset_names()}
        return get_json_ok_response(content)
    except Exception:
        return get_json_api_error_response(UnexpectedError("Unexpected error."))


def create_is_valid_endpoint(
    external_auth_url: Optional[str] = None, max_age_long: int = 0, max_age_short: int = 0
) -> Endpoint:
    async def is_valid_endpoint(request: Request) -> Response:
        try:
            dataset_name = request.query_params.get("dataset")
            logging.info(f"/is-valid, dataset={dataset_name}")
            if not are_valid_parameters([dataset_name]):
                raise MissingRequiredParameterError("Parameter 'dataset' is required")
            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(dataset_name, external_auth_url=external_auth_url, request=request)
            content = {
                "valid": is_dataset_name_valid(dataset_name),
            }
            return get_json_ok_response(content=content, max_age=max_age_long)
        except ApiCustomError as e:
            return get_json_api_error_response(error=e, max_age=max_age_short)
        except Exception:
            return get_json_api_error_response(error=UnexpectedError("Unexpected error."), max_age=max_age_short)

    return is_valid_endpoint
