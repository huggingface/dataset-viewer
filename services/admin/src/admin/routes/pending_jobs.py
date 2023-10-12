# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libapi.exceptions import ApiError, UnexpectedApiError
from libapi.utils import Endpoint, get_json_api_error_response, get_json_ok_response
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check


def create_pending_jobs_endpoint(
    processing_graph: ProcessingGraph,
    max_age: int,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Endpoint:
    async def pending_jobs_endpoint(request: Request) -> Response:
        logging.info("/pending-jobs")
        try:
            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(
                external_auth_url=external_auth_url,
                request=request,
                organization=organization,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            queue = Queue()
            return get_json_ok_response(
                {
                    processing_step.job_type: queue.get_dump_by_pending_status(job_type=processing_step.job_type)
                    for processing_step in processing_graph.get_alphabetically_ordered_processing_steps()
                },
                max_age=max_age,
            )
        except ApiError as e:
            return get_json_api_error_response(e, max_age=max_age)
        except Exception as e:
            return get_json_api_error_response(UnexpectedApiError("Unexpected error.", e), max_age=max_age)

    return pending_jobs_endpoint
