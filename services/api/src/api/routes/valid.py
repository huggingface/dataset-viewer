# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import List, Optional

from libcache.simple_cache import get_valid_datasets, get_validity_by_kind
from starlette.requests import Request
from starlette.responses import Response

from api.authentication import auth_check
from api.utils import (
    ApiCustomError,
    CacheKind,
    Endpoint,
    MissingRequiredParameterError,
    UnexpectedError,
    are_valid_parameters,
    get_json_api_error_response,
    get_json_ok_response,
)


def get_valid() -> List[str]:
    # a dataset is considered valid if:
    # - the /splits response is valid
    datasets = get_valid_datasets(kind=CacheKind.SPLITS.value)
    # - at least one of the /first-rows responses is valid
    datasets.intersection_update(get_valid_datasets(kind=CacheKind.FIRST_ROWS.value))
    # note that the list is sorted alphabetically for consistency
    return sorted(datasets)


def is_valid(dataset: str) -> bool:
    # a dataset is considered valid if:
    # - the /splits response is valid
    # - at least one of the /first-rows responses is valid
    validity_by_kind = get_validity_by_kind(dataset=dataset)
    return (
        CacheKind.SPLITS.value in validity_by_kind
        and validity_by_kind[CacheKind.SPLITS.value]
        and CacheKind.FIRST_ROWS.value in validity_by_kind
        and validity_by_kind[CacheKind.FIRST_ROWS.value]
    )


async def valid_endpoint(_: Request) -> Response:
    try:
        logging.info("/valid")
        content = {"valid": get_valid()}
        return get_json_ok_response(content)
    except Exception:
        return get_json_api_error_response(UnexpectedError("Unexpected error."))


def create_is_valid_endpoint(
    external_auth_url: Optional[str] = None, max_age_long: int = 0, max_age_short: int = 0
) -> Endpoint:
    async def is_valid_endpoint(request: Request) -> Response:
        try:
            dataset = request.query_params.get("dataset")
            logging.info(f"/is-valid, dataset={dataset}")
            if not are_valid_parameters([dataset]):
                raise MissingRequiredParameterError("Parameter 'dataset' is required")
            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(dataset, external_auth_url=external_auth_url, request=request)
            content = {
                "valid": is_valid(dataset),
            }
            return get_json_ok_response(content=content, max_age=max_age_long)
        except ApiCustomError as e:
            return get_json_api_error_response(error=e, max_age=max_age_short)
        except Exception:
            return get_json_api_error_response(error=UnexpectedError("Unexpected error."), max_age=max_age_short)

    return is_valid_endpoint
