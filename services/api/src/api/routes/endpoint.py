# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import List, Mapping, Optional, Tuple

from libcommon.dataset import get_dataset_git_revision
from libcommon.processing_graph import InputType, ProcessingGraph, ProcessingStep
from libcommon.simple_cache import (
    CACHED_RESPONSE_NOT_FOUND,
    CacheEntry,
    get_best_response,
)
from libcommon.state import Artifact, DatasetState
from libcommon.utils import Priority
from starlette.requests import Request
from starlette.responses import Response

from api.authentication import auth_check
from api.config import EndpointConfig
from api.prometheus import StepProfiler
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


def get_cache_entry_from_steps(
    processing_steps: List[ProcessingStep],
    dataset: str,
    config: Optional[str],
    split: Optional[str],
    processing_graph: ProcessingGraph,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> CacheEntry:
    """Gets the cache from the first successful step in the processing steps list.
    If no successful result is found, it will return the last one even if it's an error,
    Checks if job is still in progress by each processing step in case of no entry found.
    Raises:
        - [`libcommon.exceptions.AskAccessHubRequestError`]
          if the request to the Hub to get access to the dataset failed or timed out.
        - [`libcommon.exceptions.DatasetInfoHubRequestError`]
          if the request to the Hub to get the dataset info failed or timed out.
        - [`libcommon.exceptions.DatasetError`]
          if the dataset could not be accessed or is not supported
        - [`~utils.ResponseNotFoundError`]
          if no result is found.
        - [`~utils.ResponseNotReadyError`]
          if the response is not ready yet.

    Returns: the cached record
    """
    kinds = [processing_step.cache_kind for processing_step in processing_steps]
    best_response = get_best_response(kinds=kinds, dataset=dataset, config=config, split=split)
    if "error_code" in best_response.response and best_response.response["error_code"] == CACHED_RESPONSE_NOT_FOUND:
        # The cache is missing. Look if the job is in progress, or if it should be backfilled.
        try:
            revision = get_dataset_git_revision(
                dataset=dataset,
                hf_endpoint=hf_endpoint,
                hf_token=hf_token,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            # ^ TODO: the revision could be in the cache (new processing step)
        except Exception as e:
            raise ResponseNotFoundError("Not found.") from e
        ERROR_CODES_TO_RETRY: List[str] = []
        # ^ TODO: pass error_codes_to_retry? or set them in the processing graph?
        dataset_state = DatasetState(
            dataset=dataset,
            processing_graph=processing_graph,
            revision=revision,
            error_codes_to_retry=ERROR_CODES_TO_RETRY,
            priority=Priority.NORMAL,
            # TODO: move Priority outside from queue.py (to remove dependency to this file)
        )
        artifact_ids = [
            Artifact(
                processing_step=processing_step, dataset=dataset, revision=revision, config=config, split=split
            ).id
            for processing_step in processing_steps
        ]
        should_exist = any(
            artifact_id in dataset_state.queue_status.in_process for artifact_id in artifact_ids
        ) or any(
            f"CreateJob,{artifact_id}" in task.id for task in dataset_state.plan.tasks for artifact_id in artifact_ids
        )

        # use the opportunity to backfill if needed
        dataset_state.backfill()

        if should_exist:
            raise ResponseNotReadyError(
                "The server is busier than usual and the response is not ready yet. Please retry later."
            )
        else:
            raise ResponseNotFoundError("Not found.")
    return best_response.response


class InputTypeValidator(ABC):
    input_type: InputType = NotImplemented

    @abstractmethod
    def are_parameters_sufficient(self, dataset: Optional[str], config: Optional[str], split: Optional[str]) -> bool:
        pass

    @abstractmethod
    def get_error_message(self) -> str:
        pass

    @abstractmethod
    def get_useful_parameters(
        self, dataset: Optional[str], config: Optional[str], split: Optional[str]
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        pass

    @staticmethod
    def from_input_type(input_type: InputType) -> "InputTypeValidator":
        return (
            DatasetInputTypeValidator()
            if input_type == "dataset"
            else ConfigInputTypeValidator()
            if input_type == "config"
            else SplitInputTypeValidator()
        )


class DatasetInputTypeValidator(InputTypeValidator):
    input_type: InputType = "dataset"

    def are_parameters_sufficient(self, dataset: Optional[str], config: Optional[str], split: Optional[str]) -> bool:
        return are_valid_parameters([dataset])

    def get_error_message(self) -> str:
        return "Parameter 'dataset' is required"

    def get_useful_parameters(
        self, dataset: Optional[str], config: Optional[str], split: Optional[str]
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        return (dataset, None, None)


class ConfigInputTypeValidator(InputTypeValidator):
    input_type: InputType = "config"

    def are_parameters_sufficient(self, dataset: Optional[str], config: Optional[str], split: Optional[str]) -> bool:
        return are_valid_parameters([dataset, config])

    def get_error_message(self) -> str:
        return "Parameters 'config' and 'dataset' are required"

    def get_useful_parameters(
        self, dataset: Optional[str], config: Optional[str], split: Optional[str]
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        return (dataset, config, None)


class SplitInputTypeValidator(InputTypeValidator):
    input_type: InputType = "split"

    def are_parameters_sufficient(self, dataset: Optional[str], config: Optional[str], split: Optional[str]) -> bool:
        return are_valid_parameters([dataset, config, split])

    def get_error_message(self) -> str:
        return "Parameters 'split', 'config' and 'dataset' are required"

    def get_useful_parameters(
        self, dataset: Optional[str], config: Optional[str], split: Optional[str]
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        return (dataset, config, split)


def get_input_type_validators_by_priority(steps_by_input_type: StepsByInputType) -> List[InputTypeValidator]:
    input_type_order: List[InputType] = ["split", "config", "dataset"]
    return [
        InputTypeValidator.from_input_type(input_type)
        for input_type in input_type_order
        if input_type in steps_by_input_type
    ]


def get_input_type_validator_by_parameters(
    validators: List[InputTypeValidator], dataset: Optional[str], config: Optional[str], split: Optional[str]
) -> InputTypeValidator:
    error_message = "No processing steps supported for parameters"
    for validator in validators:
        error_message = validator.get_error_message()
        if validator.are_parameters_sufficient(dataset=dataset, config=config, split=split):
            return validator
    raise MissingRequiredParameterError(error_message)


def create_endpoint(
    endpoint_name: str,
    steps_by_input_type: StepsByInputType,
    processing_graph: ProcessingGraph,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_jwt_public_key: Optional[str] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Endpoint:
    async def processing_step_endpoint(request: Request) -> Response:
        context = f"endpoint: {endpoint_name}"
        with StepProfiler(method="processing_step_endpoint", step="all", context=context):
            try:
                with StepProfiler(
                    method="processing_step_endpoint",
                    step="validate parameters and get processing steps",
                    context=context,
                ):
                    # validating request parameters
                    dataset_parameter = request.query_params.get("dataset")
                    config_parameter = request.query_params.get("config")
                    split_parameter = request.query_params.get("split")
                    validators = get_input_type_validators_by_priority(steps_by_input_type=steps_by_input_type)

                    logging.debug(
                        f"endpoint={endpoint_name} dataset={dataset_parameter} config={config_parameter}"
                        + f" split={split_parameter}"
                    )

                    validator = get_input_type_validator_by_parameters(
                        validators, dataset_parameter, config_parameter, split_parameter
                    )
                    processing_steps = steps_by_input_type[validator.input_type]
                    dataset, config, split = validator.get_useful_parameters(
                        dataset_parameter, config_parameter, split_parameter
                    )

                # for now, dataset is always required in the endpoints.
                if not dataset:
                    raise MissingRequiredParameterError("Parameter 'dataset' is required")

                # if auth_check fails, it will raise an exception that will be caught below
                with StepProfiler(method="processing_step_endpoint", step="check authentication", context=context):
                    auth_check(
                        dataset,
                        external_auth_url=external_auth_url,
                        request=request,
                        hf_jwt_public_key=hf_jwt_public_key,
                        hf_jwt_algorithm=hf_jwt_algorithm,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )
                # getting result based on processing steps
                with StepProfiler(method="processing_step_endpoint", step="get cache entry", context=context):
                    result = get_cache_entry_from_steps(
                        processing_steps=processing_steps,
                        dataset=dataset,
                        config=config,
                        split=split,
                        processing_graph=processing_graph,
                        hf_endpoint=hf_endpoint,
                        hf_token=hf_token,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )
                content = result["content"]
                http_status = result["http_status"]
                error_code = result["error_code"]
                if http_status == HTTPStatus.OK:
                    with StepProfiler(method="processing_step_endpoint", step="generate OK response", context=context):
                        return get_json_ok_response(content=content, max_age=max_age_long)

                with StepProfiler(method="processing_step_endpoint", step="generate error response", context=context):
                    return get_json_error_response(
                        content=content, status_code=http_status, max_age=max_age_short, error_code=error_code
                    )
            except Exception as e:
                error = e if isinstance(e, ApiCustomError) else UnexpectedError("Unexpected error.", e)
                with StepProfiler(
                    method="processing_step_endpoint", step="generate API error response", context=context
                ):
                    return get_json_api_error_response(error=error, max_age=max_age_short)

    return processing_step_endpoint
