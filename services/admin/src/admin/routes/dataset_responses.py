# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.processing_graph import ProcessingGraph
from libcommon.simple_cache import get_dataset_responses_with_content_for_kind
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


def create_dataset_responses_endpoint(
    processing_graph: ProcessingGraph,
    max_age: int,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Endpoint:
    async def usage_endpoint(request: Request) -> Response:
        try:
            kind = request.query_params.get("kind")
            if not are_valid_parameters([kind]) or not kind:
                raise MissingRequiredParameterError("Parameter 'kind' is required")
            input_type = processing_graph.get_processing_step(kind).input_type
            dataset = request.query_params.get("dataset")
            config = request.query_params.get("config")
            split = request.query_params.get("split")
            logging.info(f"/trending-status, kind={kind} dataset={dataset} config={config} split={split}")
            if not are_valid_parameters([dataset]) or not dataset:
                raise MissingRequiredParameterError("Parameter 'dataset' is required")
            if input_type in ("config", "split") and not are_valid_parameters([config]) or not config:
                raise MissingRequiredParameterError("Parameter 'config' is required")
            if input_type == "split" and not are_valid_parameters([split]) or not split:
                raise MissingRequiredParameterError("Parameter 'split' is required")

            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(
                external_auth_url=external_auth_url,
                request=request,
                organization=organization,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            dataset_responses = get_dataset_responses_with_content_for_kind(
                kind, dataset=dataset, config=config, split=split
            )

            return get_json_ok_response(
                dataset_responses,
                max_age=max_age,
            )
        except AdminCustomError as e:
            return get_json_admin_error_response(e, max_age=max_age)
        except Exception as e:
            return get_json_admin_error_response(UnexpectedError("Unexpected error.", e), max_age=max_age)

    return usage_endpoint
