# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Queue
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check
from admin.utils import (
    AdminCustomError,
    Endpoint,
    UnexpectedError,
    get_json_admin_error_response,
    get_json_ok_response,
)


def create_jobs_duration_per_dataset_endpoint(
    processing_step: ProcessingStep,
    max_age: int,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
) -> Endpoint:
    async def jobs_duration_per_dataset_endpoint(request: Request) -> Response:
        logging.info("/jobs-duration-per-dataset")
        try:
            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(external_auth_url=external_auth_url, request=request, organization=organization)
            queue = Queue()
            return get_json_ok_response(
                queue.get_total_duration_per_dataset(job_type=processing_step.job_type),
                max_age=max_age,
            )
        except AdminCustomError as e:
            return get_json_admin_error_response(e, max_age=max_age)
        except Exception as e:
            return get_json_admin_error_response(UnexpectedError("Unexpected error.", e), max_age=max_age)

    return jobs_duration_per_dataset_endpoint
