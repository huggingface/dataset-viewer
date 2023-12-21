# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libapi.exceptions import UnexpectedApiError
from libapi.request import get_request_parameter
from libapi.utils import Endpoint, get_json_api_error_response, get_json_ok_response
from libcommon.exceptions import CustomError
from libcommon.operations import get_dataset_status
from libcommon.orchestrator import DatasetOrchestrator
from libcommon.utils import Priority, SupportStatus
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check


def create_dataset_backfill_endpoint(
    hf_endpoint: str,
    cache_max_days: int,
    blocked_datasets: list[str],
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Endpoint:
    async def dataset_backfill_endpoint(request: Request) -> Response:
        try:
            dataset = get_request_parameter(request, "dataset", required=True)
            logging.info(f"/dataset-backfill, dataset={dataset}")

            # if auth_check fails, it will raise an exception that will be caught below
            await auth_check(
                external_auth_url=external_auth_url,
                request=request,
                organization=organization,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            dataset_status = get_dataset_status(
                dataset=dataset,
                hf_endpoint=hf_endpoint,
                hf_token=hf_token,
                hf_timeout_seconds=hf_timeout_seconds,
                blocked_datasets=blocked_datasets,
            )
            if dataset_status.support_status == SupportStatus.UNSUPPORTED:
                raise ValueError(f"Dataset {dataset} is not supported.")
            DatasetOrchestrator(dataset=dataset).backfill(
                revision=dataset_status.revision, priority=Priority.LOW, cache_max_days=cache_max_days
            )
            return get_json_ok_response(
                {"status": "ok", "message": "Backfilling dataset."},
                max_age=0,
            )
        except CustomError as e:
            return get_json_api_error_response(e, max_age=0)
        except Exception as e:
            return get_json_api_error_response(UnexpectedApiError("Unexpected error.", e), max_age=0)

    return dataset_backfill_endpoint
