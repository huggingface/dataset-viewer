# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

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


def create_dataset_state_endpoint(
    processing_graph: ProcessingGraph,
    max_age: int,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
) -> Endpoint:
    async def dataset_state_endpoint(request: Request) -> Response:
        try:
            dataset = request.query_params.get("dataset")
            if not are_valid_parameters([dataset]) or not dataset:
                raise MissingRequiredParameterError("Parameter 'dataset' is required")
            logging.info(f"/dataset-state, dataset={dataset}")

            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(external_auth_url=external_auth_url, request=request, organization=organization)

            dataset_state = DatasetState(dataset=dataset, processing_graph=processing_graph)
            return get_json_ok_response(dataset_state.as_response(), max_age=max_age)
        except AdminCustomError as e:
            return get_json_admin_error_response(e, max_age=max_age)
        except Exception as e:
            return get_json_admin_error_response(UnexpectedError("Unexpected error.", e), max_age=max_age)

    return dataset_state_endpoint
