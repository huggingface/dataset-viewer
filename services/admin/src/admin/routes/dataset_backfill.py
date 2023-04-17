# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.dataset import DatasetError, check_support
from libcommon.processing_graph import ProcessingGraph
from libcommon.state import DatasetState
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check
from admin.utils import (
    AdminCustomError,
    Endpoint,
    MissingRequiredParameterError,
    UnexpectedError,
    are_valid_parameters,
    get_json_admin_error_response,
    get_json_ok_response,
)


def create_dataset_backfill_endpoint(
    processing_graph: ProcessingGraph,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
) -> Endpoint:
    async def dataset_backfill_endpoint(request: Request) -> Response:
        try:
            dataset = request.query_params.get("dataset")
            if not are_valid_parameters([dataset]) or not dataset:
                raise MissingRequiredParameterError("Parameter 'dataset' is required")
            logging.info(f"/dataset-backfill, dataset={dataset}")

            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(external_auth_url=external_auth_url, request=request, organization=organization)
            check_support(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token)
            dataset_state = DatasetState(dataset=dataset, processing_graph=processing_graph)
            dataset_state.backfill()
            tasks_list = ", ".join(dataset_state.plan.as_response())
            return get_json_ok_response(
                {"status": "ok", "message": f"Backfilling dataset. Tasks: {tasks_list}"},
                max_age=0,
            )
        except (DatasetError, AdminCustomError) as e:
            return get_json_admin_error_response(e, max_age=0)
        except Exception as e:
            return get_json_admin_error_response(UnexpectedError("Unexpected error.", e), max_age=0)

    return dataset_backfill_endpoint
