# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.dataset import DatasetError, check_support
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Queue
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check
from admin.utils import (
    AdminCustomError,
    Endpoint,
    MissingRequiredParameterError,
    UnexpectedError,
    are_valid_parameters,
    get_json_admin_error_response,
    get_json_ok_response,
)


def create_force_refresh_endpoint(
    processing_step: ProcessingStep,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
) -> Endpoint:
    async def force_refresh_endpoint(request: Request) -> Response:
        try:
            dataset = request.query_params.get("dataset")
            if not are_valid_parameters([dataset]) or not dataset:
                raise MissingRequiredParameterError("Parameter 'dataset' is required")
            if processing_step.input_type == "dataset":
                config = None
                split = None
            elif processing_step.input_type == "config":
                config = request.query_params.get("config")
                split = None
                if not are_valid_parameters([config]):
                    raise MissingRequiredParameterError("Parameter 'config' is required")
            else:
                config = request.query_params.get("config")
                split = request.query_params.get("split")
                if not are_valid_parameters([config, split]):
                    raise MissingRequiredParameterError("Parameters 'config' and 'split' are required")
            logging.info(
                f"/force-refresh{processing_step.job_type}, dataset={dataset}, config={config}, split={split}"
            )

            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(external_auth_url=external_auth_url, request=request, organization=organization)
            check_support(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token)
            Queue().upsert_job(
                job_type=processing_step.job_type, dataset=dataset, config=config, split=split, force=True
            )
            return get_json_ok_response(
                {"status": "ok"},
                max_age=0,
            )
        except (DatasetError, AdminCustomError) as e:
            return get_json_admin_error_response(e, max_age=0)
        except Exception as e:
            return get_json_admin_error_response(UnexpectedError("Unexpected error.", e), max_age=0)

    return force_refresh_endpoint
