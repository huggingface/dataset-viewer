# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.config import ProcessingGraphConfig
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority, Queue
from libcommon.simple_cache import get_cache_info_for_kind_minor_than_version
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


def create_refresh_outdated_cache_endpoint(
    processing_step: ProcessingStep,
    processing_graph_config: ProcessingGraphConfig,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
) -> Endpoint:
    async def refresh_outdated_cache_endpoint(request: Request) -> Response:
        try:
            logging.info(f"/refresh-outdated-cache{processing_step.cache_kind}")

            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(external_auth_url=external_auth_url, request=request, organization=organization)

            current_version = processing_graph_config.specification[processing_step.cache_kind]["version"]

            logging.info(f"refresh cache entries with worker_version<{current_version}")
            for cache_info in get_cache_info_for_kind_minor_than_version(processing_step.cache_kind, current_version):
                Queue().upsert_job(
                    job_type=processing_step.job_type,
                    dataset=cache_info["dataset"],
                    config=cache_info["config"],
                    split=cache_info["split"],
                    force=True,
                    priority=Priority.LOW,
                )

            return get_json_ok_response(
                {"status": "ok"},
                max_age=0,
            )
        except AdminCustomError as e:
            return get_json_admin_error_response(e, max_age=0)
        except Exception as e:
            return get_json_admin_error_response(UnexpectedError("Unexpected error.", e), max_age=0)

    return refresh_outdated_cache_endpoint
