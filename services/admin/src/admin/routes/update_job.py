# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import logging
from typing import Optional

from bson import ObjectId
from libapi.exceptions import (
    InvalidParameterError,
    JobNotFoundError,
    UnexpectedApiError,
)
from libapi.request import get_request_parameter
from libapi.utils import Endpoint, get_json_api_error_response, get_json_ok_response
from libcommon.constants import DEFAULT_DIFFICULTY_MAX, DEFAULT_DIFFICULTY_MIN
from libcommon.dtos import Priority
from libcommon.exceptions import CustomError
from libcommon.queue.jobs import JobDoesNotExistError, Queue
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check


def create_update_job_endpoint(
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Endpoint:
    async def update_job_endpoint(request: Request) -> Response:
        try:
            job_id = get_request_parameter(request, "job_id", required=True)
            if not ObjectId.is_valid(job_id):
                raise InvalidParameterError("Parameter 'job_id' should be a valid job id.")
            priority_parameter = get_request_parameter(request, "priority")
            difficulty_parameter = get_request_parameter(request, "difficulty")
            if not priority_parameter and not difficulty_parameter:
                raise InvalidParameterError("At least one of the parameters 'priority' and 'difficulty' is required.")
            priority: Optional[Priority] = None
            if priority_parameter:
                try:
                    priority = Priority(priority_parameter)
                except ValueError:
                    raise InvalidParameterError(
                        f"Parameter 'priority' should be one of {', '.join(prio.value for prio in Priority)}."
                    )
            difficulty: Optional[int] = None
            if difficulty_parameter:
                difficulty_error_message = (
                    f"Parameter 'difficulty' should be an int value between "
                    f"{DEFAULT_DIFFICULTY_MIN} and {DEFAULT_DIFFICULTY_MAX}."
                )
                try:
                    difficulty = int(difficulty_parameter)
                except ValueError:
                    raise InvalidParameterError(difficulty_error_message)
                if difficulty < DEFAULT_DIFFICULTY_MIN or difficulty > DEFAULT_DIFFICULTY_MAX:
                    raise InvalidParameterError(difficulty_error_message)
            logging.info(f"/update-job, {job_id=}, {priority=}, {difficulty=}")

            # if auth_check fails, it will raise an exception that will be caught below
            await auth_check(
                external_auth_url=external_auth_url,
                request=request,
                organization=organization,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            job = Queue().update_job(job_id=job_id, priority=priority, difficulty=difficulty)
            return get_json_ok_response(
                {"status": "ok", "job": job.to_dict()},
                max_age=0,
            )
        except JobDoesNotExistError as e:
            return get_json_api_error_response(
                JobNotFoundError("The job does not exist or has already been processed.", e), max_age=0
            )
        except CustomError as e:
            return get_json_api_error_response(e, max_age=0)
        except Exception as e:
            return get_json_api_error_response(UnexpectedApiError("Unexpected error.", e), max_age=0)

    return update_job_endpoint
