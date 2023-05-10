# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.dataset import DatasetError, get_dataset_git_revision
from libcommon.processing_graph import InputType
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
    input_type: InputType,
    job_type: str,
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
            if input_type == "dataset":
                config = None
                split = None
            elif input_type == "config":
                config = request.query_params.get("config")
                split = None
                if not are_valid_parameters([config]):
                    raise MissingRequiredParameterError("Parameter 'config' is required")
            else:
                config = request.query_params.get("config")
                split = request.query_params.get("split")
                if not are_valid_parameters([config, split]):
                    raise MissingRequiredParameterError("Parameters 'config' and 'split' are required")
            logging.info(f"/force-refresh{job_type}, dataset={dataset}, config={config}, split={split}")

            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(external_auth_url=external_auth_url, request=request, organization=organization)
            get_dataset_git_revision(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token)
            # ^ TODO: pass the revision to the job (meanwhile: checks if the dataset is supported)
            Queue().upsert_job(job_type=job_type, dataset=dataset, config=config, split=split)
            return get_json_ok_response(
                {"status": "ok"},
                max_age=0,
            )
        except (DatasetError, AdminCustomError) as e:
            return get_json_admin_error_response(e, max_age=0)
        except Exception as e:
            return get_json_admin_error_response(UnexpectedError("Unexpected error.", e), max_age=0)

    return force_refresh_endpoint
