# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from libcommon.processing_steps import Parameters, ProcessingStep
from libqueue.queue import Queue
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check
from admin.utils import (
    AdminCustomError,
    Endpoint,
    MissingRequiredParameterError,
    UnexpectedError,
    UnsupportedDatasetError,
    are_valid_parameters,
    get_json_admin_error_response,
    get_json_ok_response,
)


def check_support(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> None:
    """
    Check if the dataset exists on the Hub and is supported by the datasets-server.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
    Returns:
        `None`
    Raises:
        UnsupportedDatasetError: if the dataset is not supported
    """
    try:
        # note that token is required to access gated dataset info
        info = HfApi(endpoint=hf_endpoint).dataset_info(dataset, token=hf_token)
        if info.private is True:
            raise UnsupportedDatasetError(f"Dataset '{dataset}' is not supported.")
    except RepositoryNotFoundError as e:
        raise UnsupportedDatasetError(f"Dataset '{dataset}' is not supported.") from e


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
            if not are_valid_parameters([dataset]):
                raise MissingRequiredParameterError("Parameter 'dataset' is required")
            if processing_step.parameters == Parameters.DATASET:
                config = None
                split = None
            else:
                config = request.query_params.get("config")
                split = request.query_params.get("split")
                if not are_valid_parameters([config, split]):
                    raise MissingRequiredParameterError("Parameters 'config' and 'split' are required")
            logging.info(
                f"/force-refresh{processing_step.endpoint}, dataset={dataset}, config={config}, split={split}"
            )

            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(external_auth_url=external_auth_url, request=request, organization=organization)
            check_support(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token)
            Queue(type=processing_step.job_type).add_job(dataset=dataset, config=config, split=split, force=True)
            return get_json_ok_response(
                {"status": "ok"},
                max_age=0,
            )
        except AdminCustomError as e:
            return get_json_admin_error_response(e, max_age=0)
        except Exception:
            return get_json_admin_error_response(UnexpectedError("Unexpected error."), max_age=0)

    return force_refresh_endpoint
