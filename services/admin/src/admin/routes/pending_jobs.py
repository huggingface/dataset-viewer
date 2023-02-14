# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import List, Optional

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


def create_pending_jobs_endpoint(
    processing_steps: List[ProcessingStep],
    max_age: int,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
) -> Endpoint:
    async def pending_jobs_endpoint(request: Request) -> Response:
        logging.info("/pending-jobs")
        try:
            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(external_auth_url=external_auth_url, request=request, organization=organization)
            queue = Queue()
            return get_json_ok_response(
                {
                    processing_step.job_type: queue.get_dump_by_pending_status(job_type=processing_step.job_type)
                    for processing_step in processing_steps
                },
                max_age=max_age,
            )
        except AdminCustomError as e:
            return get_json_admin_error_response(e, max_age=max_age)
        except Exception as e:
            return get_json_admin_error_response(UnexpectedError("Unexpected error.", e), max_age=max_age)

    return pending_jobs_endpoint
