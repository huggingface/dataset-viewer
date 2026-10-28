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
from libcommon.exceptions import CustomError
from libcommon.queue.jobs import JobDoesNotExistError, Queue
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check


def create_delete_job_endpoint(
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Endpoint:
    async def delete_job_endpoint(request: Request) -> Response:
        try:
            job_id = get_request_parameter(request, "job_id", required=True)
            if not ObjectId.is_valid(job_id):
                raise InvalidParameterError("Parameter 'job_id' should be a valid job id.")
            logging.info(f"/delete-job, {job_id=}")

            # if auth_check fails, it will raise an exception that will be caught below
            await auth_check(
                external_auth_url=external_auth_url,
                request=request,
                organization=organization,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            job = Queue().delete_job(job_id=job_id)
            return get_json_ok_response(
                {"status": "ok", "job_id": str(job.pk)},
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

    return delete_job_endpoint
