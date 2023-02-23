# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import List, Mapping, Optional

from libcommon.dataset import DatasetError
from libcommon.operations import PreviousStepError, check_in_process
from libcommon.processing_graph import ProcessingGraph, ProcessingStep
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

InputForProcessingStepsMapping = Mapping[str, List[ProcessingStep]]

EndpointProcessingStepsMapping = Mapping[str, InputForProcessingStepsMapping]


class EndpointsDefinition:
    """Definition of supported endpoints and its relation with processing steps."""

    processing_steps_by_endpoint: EndpointProcessingStepsMapping

    def __init__(self, graph: ProcessingGraph, endpoint_config: EndpointConfig):
        self.definition = {
            endpoint: {
                input_type: [graph.get_step(step) for step in steps] for input_type, steps in input_types.items()
            }
            for endpoint, input_types in endpoint_config.specification.items()
        }


@dataclass
class InputParams:
    dataset: str
    config: Optional[str]
    split: Optional[str]
    input_type: str = field(init=False)

    def __post_init__(self) -> None:
        self.input_type = "split" if self.split else "config" if self.config else "dataset"


def get_params(query_params: QueryParams) -> InputParams:
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

    return InputParams(dataset=dataset, config=config, split=split)


def first_entry_from_steps(processing_steps: List[ProcessingStep], params: InputParams) -> Optional[CacheEntry]:
    for processing_step in processing_steps:
        try:
            result = get_response(
                kind=processing_step.cache_kind,
                dataset=params.dataset,
                config=params.config,
                split=params.split,
            )
            return result
        except DoesNotExist:
            logging.warning(
                f"processing_step={processing_step.name} dataset={params.dataset} "
                f"config={params.config} split={params.split} no entry found"
            )
    return None


def create_endpoint(
    input_types: InputForProcessingStepsMapping,
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
            params: InputParams = get_params(request.query_params)
            logging.info(
                f"input_type={params.input_type}, dataset={params.dataset}, config={params.config},"
                f" split={params.split}"
            )

            if params.input_type not in input_types:
                raise MissingRequiredParameterError("Operation not supported or not found")

            processing_steps = input_types[params.input_type]
            if not processing_steps:
                raise MissingProcessingStepsError("Missing processing steps")

            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(params.dataset, external_auth_url=external_auth_url, request=request)

            # try to get result from steps
            result = first_entry_from_steps(processing_steps, params)
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

            # maybe the response is in process"
            logging.warning("No response found for processing steps")
            for processing_step in processing_steps:
                try:
                    check_in_process(
                        processing_step=processing_step,
                        init_processing_steps=init_processing_steps,
                        dataset=params.dataset,
                        config=params.config,
                        split=params.split,
                        hf_endpoint=hf_endpoint,
                        hf_token=hf_token,
                    )
                except (PreviousStepError, DatasetError):
                    raise ResponseNotFoundError("Not found.")
                raise ResponseNotReadyError(
                    "The server is busier than usual and the response is not ready yet. Please retry later."
                )
        except ApiCustomError as e:
            return get_json_api_error_response(error=e, max_age=max_age_short)
        except Exception as e:
            return get_json_api_error_response(error=UnexpectedError("Unexpected error.", e), max_age=max_age_short)

    return processing_step_endpoint
