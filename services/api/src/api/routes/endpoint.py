# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import List, Mapping, Optional

from libcommon.dataset import DatasetError
from libcommon.operations import PreviousStepError, check_in_process
from libcommon.processing_graph import InputType, ProcessingGraph, ProcessingStep
from libcommon.simple_cache import CacheEntry, DoesNotExist, get_response
from starlette.requests import Request
from starlette.responses import Response

from api.authentication import auth_check
from api.config import EndpointConfig
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

StepsByInputType = Mapping[InputType, List[ProcessingStep]]

StepsByInputTypeAndEndpoint = Mapping[str, StepsByInputType]


class EndpointsDefinition:
    """Definition of supported endpoints and its relation with processing steps."""

    steps_by_input_type_and_endpoint: StepsByInputTypeAndEndpoint

    def __init__(self, graph: ProcessingGraph, endpoint_config: EndpointConfig):
        self.steps_by_input_type_and_endpoint = {
            endpoint: {
                input_type: [graph.get_step(step_name) for step_name in step_names]
                for input_type, step_names in step_names_by_input_type.items()
            }
            for endpoint, step_names_by_input_type in endpoint_config.step_names_by_input_type_and_endpoint.items()
        }


def get_first_succeeded_cache_entry_from_steps(
    processing_steps: List[ProcessingStep],
    dataset: str,
    config: Optional[str],
    split: Optional[str],
    init_processing_steps: List[ProcessingStep],
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> Optional[CacheEntry]:
    """Gets the cache from the first succedeed step in the processing step list.
    If no succeeded result is found, it will return the last one even if it was failed,
    if no one result is found, it returns None.
    Checks if job is still in progress by each processing step in case of no entry found.
    """
    last_result = None
    for processing_step in processing_steps:
        try:
            last_result = get_response(
                kind=processing_step.cache_kind,
                dataset=dataset,
                config=config,
                split=split,
            )

            if last_result["http_status"] == HTTPStatus.OK:
                return last_result
        except DoesNotExist:
            logging.debug(
                f"processing_step={processing_step.name} dataset={dataset} "
                f"config={config} split={split} no entry found"
            )
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
                raise ResponseNotFoundError("Not found.")
    return last_result


def get_response_from_cache_entry(
    result: Optional[CacheEntry],
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Response:
    if not result:
        raise ResponseNotReadyError(
            "The server is busier than usual and the response is not ready yet. Please retry later."
        )

    content = result["content"]
    http_status = result["http_status"]
    error_code = result["error_code"]
    if http_status == HTTPStatus.OK:
        return get_json_ok_response(content=content, max_age=max_age_long)

    return get_json_error_response(
        content=content, status_code=http_status, max_age=max_age_short, error_code=error_code
    )


def get_input_types_by_priority(steps_by_input_type: StepsByInputType) -> List[InputType]:
    input_type_order: List[InputType] = ["split", "config", "dataset"]
    return [input_type for input_type in input_type_order if input_type in steps_by_input_type]


def are_request_parameters_enough_for_input_type(
    input_type: InputType, dataset: Optional[str], config: Optional[str], split: Optional[str]
) -> bool:
    parameters = (
        [dataset]
        if input_type == "dataset"
        else [config, dataset]
        if input_type == "config"
        else [split, config, dataset]
    )
    return are_valid_parameters(parameters)


def create_endpoint(
    endpoint_name: str,
    steps_by_input_type: StepsByInputType,
    init_processing_steps: List[ProcessingStep],
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Endpoint:
    async def processing_step_endpoint(request: Request) -> Response:
        try:
            # validating request parameters
            dataset = request.query_params.get("dataset")

            if not are_valid_parameters([dataset]) or not dataset:
                raise MissingRequiredParameterError("Parameter 'dataset' is required")

            config = request.query_params.get("config")
            split = request.query_params.get("split")

            input_types = get_input_types_by_priority(steps_by_input_type=steps_by_input_type)

            logging.debug(f"endpoint={endpoint_name} dataset={dataset} config={config} split={split}")
            processing_steps = []
            error_message = "Parameter 'dataset' is required"
            for input_type in input_types:
                if are_request_parameters_enough_for_input_type(input_type, dataset, config, split):
                    processing_steps = steps_by_input_type[input_type]
                    config = None if input_type == "dataset" else config
                    split = None if input_type in ("dataset", "config") else split
                    logging.debug(f"Input type = {input_type} is the appropiated for the params")
                    break
                elif input_type == "config":
                    error_message = "Parameters 'config' and 'dataset' are required"
                elif input_type == "split":
                    error_message = "Parameters 'split', 'config' and 'dataset' are required"

            if not processing_steps:
                raise MissingRequiredParameterError(error_message)

            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(dataset, external_auth_url=external_auth_url, request=request)

            # getting result based on processing steps
            result = get_first_succeeded_cache_entry_from_steps(
                processing_steps, dataset, config, split, init_processing_steps, hf_endpoint, hf_token
            )

            return get_response_from_cache_entry(result, max_age_long, max_age_short)
        except ApiCustomError as e:
            return get_json_api_error_response(error=e, max_age=max_age_short)
        except Exception as e:
            return get_json_api_error_response(error=UnexpectedError("Unexpected error.", e), max_age=max_age_short)

    return processing_step_endpoint
