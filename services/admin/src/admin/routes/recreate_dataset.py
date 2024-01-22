# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from typing import Optional, TypedDict

from libapi.exceptions import InvalidParameterError, UnexpectedApiError
from libapi.request import get_request_parameter
from libapi.utils import Endpoint, get_json_api_error_response, get_json_ok_response
from libcommon.dtos import Priority
from libcommon.exceptions import CustomError
from libcommon.operations import delete_dataset, update_dataset
from libcommon.storage_client import StorageClient
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check


class RecreateDatasetReport(TypedDict):
    status: str
    dataset: str


def recreate_dataset(
    dataset: str,
    priority: Priority,
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    storage_clients: Optional[list[StorageClient]] = None,
) -> RecreateDatasetReport:
    delete_dataset(dataset=dataset, storage_clients=storage_clients)
    # create the jobs to backfill the dataset, if supported
    update_dataset(
        dataset=dataset,
        blocked_datasets=blocked_datasets,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
        priority=priority,
        storage_clients=storage_clients,
    )
    return RecreateDatasetReport(status="ok", dataset=dataset)


def create_recreate_dataset_endpoint(
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    storage_clients: Optional[list[StorageClient]] = None,
) -> Endpoint:
    async def recreate_dataset_endpoint(request: Request) -> Response:
        try:
            dataset = get_request_parameter(request, "dataset", required=True)
            try:
                priority = Priority(get_request_parameter(request, "priority", default="low"))
            except ValueError:
                raise InvalidParameterError(
                    f"Parameter 'priority' should be one of {', '.join(prio.value for prio in Priority)}."
                )
            logging.info(f"/recreate-dataset, dataset={dataset}, priority={priority}")

            # if auth_check fails, it will raise an exception that will be caught below
            await auth_check(
                external_auth_url=external_auth_url,
                request=request,
                organization=organization,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            return get_json_ok_response(
                recreate_dataset(
                    dataset=dataset,
                    priority=priority,
                    blocked_datasets=blocked_datasets,
                    hf_endpoint=hf_endpoint,
                    hf_token=hf_token,
                    storage_clients=storage_clients,
                ),
                max_age=0,
            )
        except CustomError as e:
            return get_json_api_error_response(e, max_age=0)
        except Exception as e:
            return get_json_api_error_response(UnexpectedApiError("Unexpected error.", e), max_age=0)

    return recreate_dataset_endpoint
