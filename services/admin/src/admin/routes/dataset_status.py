# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libapi.exceptions import ApiError, UnexpectedApiError
from libapi.request import get_required_request_parameter
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue
from libcommon.simple_cache import get_dataset_responses_without_content_for_kind
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check
from admin.utils import Endpoint, get_json_admin_error_response, get_json_ok_response


def create_dataset_status_endpoint(
    processing_graph: ProcessingGraph,
    max_age: int,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Endpoint:
    async def dataset_status_endpoint(request: Request) -> Response:
        try:
            dataset = get_required_request_parameter(request, "dataset")
            logging.info(f"/dataset-status, dataset={dataset}")

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
                    processing_step.name: {
                        "cached_responses": get_dataset_responses_without_content_for_kind(
                            kind=processing_step.cache_kind, dataset=dataset
                        ),
                        "jobs": queue.get_dataset_pending_jobs_for_type(
                            dataset=dataset, job_type=processing_step.job_type
                        ),
                    }
                    for processing_step in processing_graph.get_alphabetically_ordered_processing_steps()
                },
                max_age=max_age,
            )
        except ApiError as e:
            return get_json_admin_error_response(e, max_age=max_age)
        except Exception as e:
            return get_json_admin_error_response(UnexpectedApiError("Unexpected error.", e), max_age=max_age)

    return dataset_status_endpoint
