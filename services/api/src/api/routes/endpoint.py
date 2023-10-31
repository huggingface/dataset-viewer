# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from collections.abc import Mapping
from http import HTTPStatus
from typing import Optional, TypedDict

from libapi.authentication import auth_check
from libapi.exceptions import (
    ApiError,
    MissingRequiredParameterError,
    UnexpectedApiError,
)
from libapi.request import get_request_parameter
from libapi.utils import (
    Endpoint,
    are_valid_parameters,
    get_cache_entry_from_steps,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
)
from libcommon.processing_graph import InputType, ProcessingGraph, ProcessingStep
from libcommon.prometheus import StepProfiler
from starlette.requests import Request
from starlette.responses import Response

from api.config import EndpointConfig

StepsByInputType = Mapping[InputType, list[ProcessingStep]]

StepsByInputTypeAndEndpoint = Mapping[str, StepsByInputType]


class EndpointsDefinition:
    """Definition of supported endpoints and its relation with processing steps."""

    steps_by_input_type_and_endpoint: StepsByInputTypeAndEndpoint

    def __init__(self, graph: ProcessingGraph, endpoint_config: EndpointConfig):
        processing_step_names_by_input_type_and_endpoint = (
            endpoint_config.processing_step_names_by_input_type_and_endpoint.items()
        )
        self.steps_by_input_type_and_endpoint = {
            endpoint: {
                input_type: [
                    graph.get_processing_step(processing_step_name) for processing_step_name in processing_step_names
                ]
                for input_type, processing_step_names in processing_step_names_by_input_type.items()
            }
            for endpoint, processing_step_names_by_input_type in processing_step_names_by_input_type_and_endpoint
        }


# TODO: remove once full scan is implemented for spawning urls scan
class OptInOutUrlsCountResponse(TypedDict):
    urls_columns: list[str]
    num_opt_in_urls: int
    num_opt_out_urls: int
    num_urls: int
    num_scanned_rows: int
    has_urls_columns: bool
    full_scan: Optional[bool]


# TODO: remove once full scan is implemented for spawning urls scan
HARD_CODED_OPT_IN_OUT_URLS = {
    "laion/laion2B-en": OptInOutUrlsCountResponse(
        urls_columns=["URL"],
        num_opt_in_urls=5,
        num_opt_out_urls=42785281,
        num_urls=2322161807,
        num_scanned_rows=0,  # It is unknown but leaving with 0 for now since UI validates non null
        has_urls_columns=True,
        full_scan=True,
    ),
    "kakaobrain/coyo-700m": OptInOutUrlsCountResponse(
        urls_columns=["url"],
        num_opt_in_urls=2,
        num_opt_out_urls=4691511,
        num_urls=746972269,
        num_scanned_rows=0,  # It is unknown but leaving with 0 for now since UI validates non null
        has_urls_columns=True,
        full_scan=True,
    ),
}


def get_input_types_by_priority(steps_by_input_type: StepsByInputType) -> list[InputType]:
    input_type_order: list[InputType] = ["split", "config", "dataset"]
    return [input_type for input_type in input_type_order if input_type in steps_by_input_type]


def create_endpoint(
    endpoint_name: str,
    steps_by_input_type: StepsByInputType,
    processing_graph: ProcessingGraph,
    cache_max_days: int,
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    hf_jwt_public_keys: Optional[list[str]] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Endpoint:
    async def processing_step_endpoint(request: Request) -> Response:
        context = f"endpoint: {endpoint_name}"
        revision: Optional[str] = None
        with StepProfiler(method="processing_step_endpoint", step="all", context=context):
            try:
                with StepProfiler(
                    method="processing_step_endpoint",
                    step="validate parameters and get processing steps",
                    context=context,
                ):
                    dataset = get_request_parameter(request, "dataset")
                    config = get_request_parameter(request, "config") or None
                    split = get_request_parameter(request, "split") or None
                    logging.debug(f"endpoint={endpoint_name} dataset={dataset} config={config} split={split}")
                    dataset, config, split, input_type = validate_parameters(
                        dataset, config, split, steps_by_input_type
                    )
                    processing_steps = steps_by_input_type[input_type]
                # if auth_check fails, it will raise an exception that will be caught below
                with StepProfiler(method="processing_step_endpoint", step="check authentication", context=context):
                    await auth_check(
                        dataset,
                        external_auth_url=external_auth_url,
                        request=request,
                        hf_jwt_public_keys=hf_jwt_public_keys,
                        hf_jwt_algorithm=hf_jwt_algorithm,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )
                # getting result based on processing steps
                with StepProfiler(method="processing_step_endpoint", step="get cache entry", context=context):
                    # TODO: remove once full scan is implemented for spawning urls scan
                    if (
                        endpoint_name == "/opt-in-out-urls"
                        and input_type == "dataset"
                        and dataset in HARD_CODED_OPT_IN_OUT_URLS
                    ):
                        return get_json_ok_response(
                            content=HARD_CODED_OPT_IN_OUT_URLS[dataset], max_age=max_age_long, revision=revision
                        )

                    result = get_cache_entry_from_steps(
                        processing_steps=processing_steps,
                        dataset=dataset,
                        config=config,
                        split=split,
                        processing_graph=processing_graph,
                        hf_endpoint=hf_endpoint,
                        hf_token=hf_token,
                        blocked_datasets=blocked_datasets,
                        hf_timeout_seconds=hf_timeout_seconds,
                        cache_max_days=cache_max_days,
                    )
                content = result["content"]
                http_status = result["http_status"]
                error_code = result["error_code"]
                revision = result["dataset_git_revision"]
                if http_status == HTTPStatus.OK:
                    with StepProfiler(method="processing_step_endpoint", step="generate OK response", context=context):
                        return get_json_ok_response(content=content, max_age=max_age_long, revision=revision)

                with StepProfiler(method="processing_step_endpoint", step="generate error response", context=context):
                    return get_json_error_response(
                        content=content,
                        status_code=http_status,
                        max_age=max_age_short,
                        error_code=error_code,
                        revision=revision,
                    )
            except Exception as e:
                error = e if isinstance(e, ApiError) else UnexpectedApiError("Unexpected error.", e)
                with StepProfiler(
                    method="processing_step_endpoint", step="generate API error response", context=context
                ):
                    return get_json_api_error_response(error=error, max_age=max_age_short, revision=revision)

    return processing_step_endpoint


def validate_parameters(
    dataset: str, config: Optional[str], split: Optional[str], steps_by_input_type: StepsByInputType
) -> tuple[str, Optional[str], Optional[str], InputType]:
    input_types = get_input_types_by_priority(steps_by_input_type=steps_by_input_type)
    error_message = "No processing steps supported for parameters"
    for input_type in input_types:
        if input_type == "split":
            if are_valid_parameters([dataset, config, split]):
                return dataset, config, split, input_type
            else:
                error_message = "Parameters 'dataset', 'config' and 'split' are required"
        elif input_type == "config":
            if are_valid_parameters([dataset, config]):
                return dataset, config, None, input_type
            else:
                error_message = "Parameters 'dataset' and 'config' are required"
        elif input_type == "dataset":
            if are_valid_parameters([dataset]):
                return dataset, None, None, input_type
            else:
                error_message = "Parameter 'dataset' is required"
    raise MissingRequiredParameterError(error_message)
