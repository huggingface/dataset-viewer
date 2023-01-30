# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import List, Optional

from libcommon.dataset import DatasetError
from libcommon.operations import PreviousStepError, check_in_process
from libcommon.processing_graph import ProcessingStep
from libcommon.simple_cache import DoesNotExist, get_response
from starlette.requests import Request
from starlette.responses import Response

from api.authentication import auth_check
from api.utils import (
    ApiCustomError,
    Endpoint,
    MissingRequiredParameterError,
    ResponseNotFoundError,
    ResponseNotReadyError,
    UnexpectedError,
    are_valid_parameters,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
)


def create_processing_step_endpoint(
    processing_step: ProcessingStep,
    init_processing_steps: List[ProcessingStep],
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Endpoint:
    async def processing_step_endpoint(request: Request) -> Response:
        try:
            dataset = request.query_params.get("dataset")
            if not are_valid_parameters([dataset]):
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
            logging.info(f"{processing_step.endpoint}, dataset={dataset}, config={config}, split={split}")

            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(dataset, external_auth_url=external_auth_url, request=request)
            try:
                result = get_response(kind=processing_step.cache_kind, dataset=dataset, config=config, split=split)
                content = result["content"]
                http_status = result["http_status"]
                error_code = result["error_code"]
                if http_status == HTTPStatus.OK:
                    return get_json_ok_response(content=content, max_age=max_age_long)
                else:
                    return get_json_error_response(
                        content=content, status_code=http_status, max_age=max_age_short, error_code=error_code
                    )
            except DoesNotExist as e:
                # maybe the response is in process
                try:
                    check_in_process(
                        processing_step=processing_step,
                        init_processing_steps=init_processing_steps,
                        dataset=dataset,
                        config=config,
                        split=split,
                        hf_endpoint=hf_endpoint,
                        hf_token=hf_token,
                    )
                except (PreviousStepError, DatasetError):
                    raise ResponseNotFoundError("Not found.") from e
                raise ResponseNotReadyError("The response is not ready yet. Please retry later.") from e
        except ApiCustomError as e:
            return get_json_api_error_response(error=e, max_age=max_age_short)
        except Exception as e:
            return get_json_api_error_response(error=UnexpectedError("Unexpected error.", e), max_age=max_age_short)

    return processing_step_endpoint
