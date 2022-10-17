# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libqueue.queue import Queue
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check
from admin.utils import (
    AdminCustomError,
    Endpoint,
    JobType,
    UnexpectedError,
    get_json_admin_error_response,
    get_json_ok_response,
)

logger = logging.getLogger(__name__)


def create_pending_jobs_endpoint(
    external_auth_url: Optional[str] = None, organization: Optional[str] = None
) -> Endpoint:
    splits_queue = Queue(type=JobType.SPLITS.value)
    first_rows_queue = Queue(type=JobType.FIRST_ROWS.value)

    async def pending_jobs_endpoint(request: Request) -> Response:
        logger.info("/pending-jobs")
        try:
            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(external_auth_url=external_auth_url, request=request, organization=organization)
            return get_json_ok_response(
                {
                    JobType.SPLITS.value: splits_queue.get_dump_by_pending_status(),
                    JobType.FIRST_ROWS.value: first_rows_queue.get_dump_by_pending_status(),
                }
            )
        except AdminCustomError as e:
            return get_json_admin_error_response(e)
        except Exception:
            return get_json_admin_error_response(UnexpectedError("Unexpected error."))

    return pending_jobs_endpoint
