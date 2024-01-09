# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libapi.exceptions import (
    InvalidParameterError,
    MissingRequiredParameterError,
    UnexpectedApiError,
)
from libapi.request import get_request_parameter
from libapi.utils import (
    Endpoint,
    are_valid_parameters,
    get_json_api_error_response,
    get_json_ok_response,
)
from libcommon.constants import MIN_BYTES_FOR_BONUS_DIFFICULTY
from libcommon.exceptions import CustomError
from libcommon.operations import get_dataset_revision_if_supported_or_raise
from libcommon.orchestrator import get_num_bytes_from_config_infos
from libcommon.processing_graph import InputType
from libcommon.queue import Queue
from libcommon.utils import Priority
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check


def create_force_refresh_endpoint(
    input_type: InputType,
    job_type: str,
    difficulty: int,
    bonus_difficulty_if_dataset_is_big: int,
    blocked_datasets: list[str],
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Endpoint:
    async def force_refresh_endpoint(request: Request) -> Response:
        try:
            dataset = get_request_parameter(request, "dataset", required=True)
            if input_type == "dataset":
                config = None
                split = None
            elif input_type == "config":
                config = get_request_parameter(request, "config", required=True)
                split = None
            else:
                config = get_request_parameter(request, "config", required=True)
                split = get_request_parameter(request, "split", required=True)
                if not are_valid_parameters([config, split]):
                    raise MissingRequiredParameterError("Parameters 'config' and 'split' are required")
            try:
                priority = Priority(get_request_parameter(request, "priority", default="low"))
            except ValueError:
                raise InvalidParameterError(
                    f"Parameter 'priority' should be one of {', '.join(prio.value for prio in Priority)}."
                )
            logging.info(
                f"/force-refresh/{job_type}, dataset={dataset}, config={config}, split={split}, priority={priority}"
            )

            total_difficulty = difficulty
            if config is not None:
                num_bytes = get_num_bytes_from_config_infos(dataset=dataset, config=config, split=split)
                if num_bytes is not None and num_bytes > MIN_BYTES_FOR_BONUS_DIFFICULTY:
                    total_difficulty += bonus_difficulty_if_dataset_is_big

            # if auth_check fails, it will raise an exception that will be caught below
            await auth_check(
                external_auth_url=external_auth_url,
                request=request,
                organization=organization,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            revision = get_dataset_revision_if_supported_or_raise(
                dataset=dataset,
                hf_endpoint=hf_endpoint,
                hf_token=hf_token,
                hf_timeout_seconds=hf_timeout_seconds,
                blocked_datasets=blocked_datasets,
            )
            Queue().add_job(
                job_type=job_type,
                difficulty=total_difficulty,
                dataset=dataset,
                revision=revision,
                config=config,
                split=split,
                priority=priority,
            )
            return get_json_ok_response(
                {"status": "ok"},
                max_age=0,
            )
        except CustomError as e:
            return get_json_api_error_response(e, max_age=0)
        except Exception as e:
            return get_json_api_error_response(UnexpectedApiError("Unexpected error.", e), max_age=0)

    return force_refresh_endpoint
