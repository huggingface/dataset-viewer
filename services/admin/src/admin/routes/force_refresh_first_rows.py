# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check
from admin.dataset import is_supported, update_first_rows
from admin.utils import (
    AdminCustomError,
    Endpoint,
    MissingRequiredParameterError,
    UnexpectedError,
    UnsupportedDatasetError,
    are_valid_parameters,
    get_json_admin_error_response,
    get_json_ok_response,
)


def create_force_refresh_first_rows_endpoint(
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
) -> Endpoint:
    async def force_refresh_first_rows_endpoint(request: Request) -> Response:
        try:
            dataset = request.query_params.get("dataset")
            config = request.query_params.get("config")
            split = request.query_params.get("split")
            logging.info(f"/force-refresh/first-rows, dataset={dataset}, config={config}, split={split}")

            if not are_valid_parameters([dataset, config, split]):
                raise MissingRequiredParameterError("Parameters 'dataset', 'config' and 'split' are required")
            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(external_auth_url=external_auth_url, request=request, organization=organization)
            if not is_supported(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token):
                raise UnsupportedDatasetError(f"Dataset '{dataset}' is not supported.")
            update_first_rows(dataset=dataset, config=config, split=split, force=True)
            return get_json_ok_response(
                {"status": "ok"},
                max_age=0,
            )
        except AdminCustomError as e:
            return get_json_admin_error_response(e, max_age=0)
        except Exception:
            return get_json_admin_error_response(UnexpectedError("Unexpected error."), max_age=0)

    return force_refresh_first_rows_endpoint
