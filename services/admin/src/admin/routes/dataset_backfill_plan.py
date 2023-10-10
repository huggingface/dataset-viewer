# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libapi.exceptions import (
    ApiError,
    MissingRequiredParameterError,
    UnexpectedApiError,
)
from libcommon.dataset import get_dataset_git_revision
from libcommon.orchestrator import DatasetBackfillPlan
from libcommon.processing_graph import ProcessingGraph
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check
from admin.utils import (
    Endpoint,
    are_valid_parameters,
    get_json_admin_error_response,
    get_json_ok_response,
)


def create_dataset_backfill_plan_endpoint(
    processing_graph: ProcessingGraph,
    max_age: int,
    hf_endpoint: str,
    cache_max_days: int,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Endpoint:
    async def dataset_state_endpoint(request: Request) -> Response:
        try:
            dataset = request.query_params.get("dataset")
            if not are_valid_parameters([dataset]) or not dataset:
                raise MissingRequiredParameterError("Parameter 'dataset' is required")
            logging.info(f"/dataset-state, dataset={dataset}")

            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(
                external_auth_url=external_auth_url,
                request=request,
                organization=organization,
                hf_timeout_seconds=hf_timeout_seconds,
            )

            dataset_git_revision = get_dataset_git_revision(
                dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, hf_timeout_seconds=hf_timeout_seconds
            )
            dataset_backfill_plan = DatasetBackfillPlan(
                dataset=dataset,
                processing_graph=processing_graph,
                revision=dataset_git_revision,
                cache_max_days=cache_max_days,
            )
            return get_json_ok_response(dataset_backfill_plan.as_response(), max_age=max_age)
        except ApiError as e:
            return get_json_admin_error_response(e, max_age=max_age)
        except Exception as e:
            return get_json_admin_error_response(UnexpectedApiError("Unexpected error.", e), max_age=max_age)

    return dataset_state_endpoint
