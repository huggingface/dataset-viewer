# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import List, Mapping, Optional

from libcommon.dataset import DatasetError
from libcommon.operations import PreviousStepError, check_in_process
from libcommon.processing_graph import InputType, ProcessingGraph, ProcessingStep
from libcommon.simple_cache import CacheEntry, DoesNotExist, get_response
from starlette.datastructures import QueryParams
from starlette.requests import Request
from starlette.responses import Response

from api.authentication import auth_check
from api.config import EndpointConfig
from api.utils import (
    ApiCustomError,
    Endpoint,
    MissingProcessingStepsError,
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

    processing_steps_by_endpoint: StepsByInputTypeAndEndpoint

    def __init__(self, graph: ProcessingGraph, endpoint_config: EndpointConfig):
        self.steps_by_input_type_and_endpoint = {
            endpoint: {
                input_type: [graph.get_step(step_name) for step_name in step_names] for input_type, step_names in step_names_by_input_type.items()
            }
            for endpoint, step_names_by_input_type in endpoint_config.step_names_by_input_type_and_endpoint.items()
        }


@dataclass
class InputParameters:
    dataset: str
    config: Optional[str]
    split: Optional[str]
    input_type: InputType = field(init=False)

    def __post_init__(self) -> None:
        self.input_type = "split" if self.split else "config" if self.config else "dataset"


def get_input_parameters(query_params: QueryParams) -> InputParameters:
    dataset = query_params.get("dataset")

    if not are_valid_parameters([dataset]) or not dataset:
        raise MissingRequiredParameterError("Parameter 'dataset' is required")

    config = None
    config_param = query_params.get("config")
    if are_valid_parameters([config_param]):
        config = config_param

    split = None
    split_param = query_params.get("split")
    if are_valid_parameters([split_param]):
        split = split_param
        if config is None:
            raise MissingRequiredParameterError("Parameter 'config' is required")

    return InputParameters(dataset=dataset, config=config, split=split)


def get_first_succeeded_cache_entry_from_steps(
    processing_steps: List[ProcessingStep],
    input_parameters: InputParameters,
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
                dataset=input_parameters.dataset,
                config=input_parameters.config,
                split=input_parameters.split,
            )

            if last_result["http_status"] == HTTPStatus.OK:
                return last_result
        except DoesNotExist:
            logging.warning(
                f"processing_step={processing_step.name} dataset={input_parameters.dataset} "
                f"config={input_parameters.config} split={input_parameters.split} no entry found"
            )
            try:
                check_in_process(
                    processing_step=processing_step,
                    init_processing_steps=init_processing_steps,
                    dataset=input_parameters.dataset,
                    config=input_parameters.config,
                    split=input_parameters.split,
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
    if result:
        content = result["content"]
        http_status = result["http_status"]
        error_code = result["error_code"]
        if http_status == HTTPStatus.OK:
            return get_json_ok_response(content=content, max_age=max_age_long)
        else:
            return get_json_error_response(
                content=content, status_code=http_status, max_age=max_age_short, error_code=error_code
            )
    raise ResponseNotReadyError(
        "The server is busier than usual and the response is not ready yet. Please retry later."
    )


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
            # getting input params
            params: InputParameters = get_input_parameters(request.query_params)
            logging.info(
                f"input_type={params.input_type}, dataset={params.dataset}, config={params.config},"
                f" split={params.split}"
            )

            if params.input_type not in steps_by_input_type:
                raise MissingRequiredParameterError("Operation not supported or not found")

            processing_steps = steps_by_input_type[params.input_type]
            if not processing_steps:
                raise MissingProcessingStepsError(
                    f"No processing steps found for endpoint '{endpoint_name}' and input type '{params.input_type}'"
                )

            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(params.dataset, external_auth_url=external_auth_url, request=request)

            result = get_first_succeeded_cache_entry_from_steps(
                processing_steps, params, init_processing_steps, hf_endpoint, hf_token
            )

            return get_response_from_cache_entry(result, max_age_long, max_age_short)
        except ApiCustomError as e:
            return get_json_api_error_response(error=e, max_age=max_age_short)
        except Exception as e:
            return get_json_api_error_response(error=UnexpectedError("Unexpected error.", e), max_age=max_age_short)

    return processing_step_endpoint
