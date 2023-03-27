# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import List, Optional, Set

from libcommon.processing_graph import ProcessingStep
from libcommon.simple_cache import get_valid_datasets, get_validity_by_kind
from starlette.requests import Request
from starlette.responses import Response

from api.authentication import auth_check
from api.prometheus import StepProfiler
from api.utils import (
    ApiCustomError,
    Endpoint,
    MissingRequiredParameterError,
    UnexpectedError,
    are_valid_parameters,
    get_json_api_error_response,
    get_json_ok_response,
)


def get_valid(processing_steps_for_valid: List[ProcessingStep]) -> List[str]:
    # a dataset is considered valid if at least one response for PROCESSING_STEPS_FOR_VALID
    # is valid.
    datasets: Optional[Set[str]] = None
    for processing_step in processing_steps_for_valid:
        kind_datasets = get_valid_datasets(kind=processing_step.cache_kind)
        if datasets is None:
            datasets = kind_datasets
        else:
            datasets.intersection_update(kind_datasets)
    # note that the list is sorted alphabetically for consistency
    return [] if datasets is None else sorted(datasets)


def is_valid(dataset: str, processing_steps_for_valid: List[ProcessingStep]) -> bool:
    # a dataset is considered valid if at least one response for PROCESSING_STEPS_FOR_VALID
    # is valid
    validity_by_kind = get_validity_by_kind(dataset=dataset)
    return all(
        processing_step.cache_kind in validity_by_kind and validity_by_kind[processing_step.cache_kind]
        for processing_step in processing_steps_for_valid
    )


def create_valid_endpoint(
    processing_steps_for_valid: List[ProcessingStep],
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Endpoint:
    # this endpoint is used by the frontend to know which datasets support the dataset viewer
    async def valid_endpoint(_: Request) -> Response:
        with StepProfiler(method="valid_endpoint", step="all"):
            try:
                logging.info("/valid")
                with StepProfiler(method="valid_endpoint", step="prepare content"):
                    content = {"valid": get_valid(processing_steps_for_valid=processing_steps_for_valid)}
                with StepProfiler(method="valid_endpoint", step="generate OK response"):
                    return get_json_ok_response(content, max_age=max_age_long)
            except Exception as e:
                with StepProfiler(method="valid_endpoint", step="generate API error response"):
                    return get_json_api_error_response(UnexpectedError("Unexpected error.", e), max_age=max_age_short)

    return valid_endpoint


def create_is_valid_endpoint(
    processing_steps_for_valid: List[ProcessingStep],
    hf_jwt_public_key: Optional[str] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Endpoint:
    # this endpoint is used to know if a dataset supports the dataset viewer
    async def is_valid_endpoint(request: Request) -> Response:
        with StepProfiler(method="is_valid_endpoint", step="all"):
            try:
                with StepProfiler(method="is_valid_endpoint", step="validate parameters and get processing steps"):
                    dataset = request.query_params.get("dataset")
                    logging.info(f"/is-valid, dataset={dataset}")
                    if not are_valid_parameters([dataset]) or not dataset:
                        raise MissingRequiredParameterError("Parameter 'dataset' is required")
                # if auth_check fails, it will raise an exception that will be caught below
                with StepProfiler(method="is_valid_endpoint", step="check authentication"):
                    auth_check(
                        dataset,
                        external_auth_url=external_auth_url,
                        request=request,
                        hf_jwt_public_key=hf_jwt_public_key,
                        hf_jwt_algorithm=hf_jwt_algorithm,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )
                with StepProfiler(method="is_valid_endpoint", step="prepare content"):
                    content = {
                        "valid": is_valid(dataset=dataset, processing_steps_for_valid=processing_steps_for_valid),
                    }
                with StepProfiler(method="is_valid_endpoint", step="generate OK response"):
                    return get_json_ok_response(content=content, max_age=max_age_long)
            except Exception as e:
                error = e if isinstance(e, ApiCustomError) else UnexpectedError("Unexpected error.", e)
                with StepProfiler(method="is_valid_endpoint", step="generate API error response"):
                    return get_json_api_error_response(error=error, max_age=max_age_short)

    return is_valid_endpoint
