# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from typing import Optional, TypedDict

from libapi.exceptions import InvalidParameterError, UnexpectedApiError
from libapi.request import get_request_parameter
from libapi.utils import Endpoint, get_json_api_error_response, get_json_ok_response
from libcommon.dataset import get_dataset_git_revision
from libcommon.exceptions import CustomError
from libcommon.operations import backfill_dataset
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue
from libcommon.simple_cache import delete_dataset_responses
from libcommon.storage_client import StorageClient
from libcommon.utils import Priority
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check


class RecreateDatasetReport(TypedDict):
    status: str
    dataset: str
    cancelled_jobs: int
    deleted_cached_responses: int


def recreate_dataset(
    dataset: str,
    priority: Priority,
    processing_graph: ProcessingGraph,
    cached_assets_storage_client: StorageClient,
    assets_storage_client: StorageClient,
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
) -> RecreateDatasetReport:
    # try to get the revision of the dataset (before deleting the jobs and the cache, in case it fails)
    revision = get_dataset_git_revision(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token)
    # delete all the jobs and all the cache entries for the dataset (for all the revisions)
    cancelled_jobs = Queue().cancel_dataset_jobs(dataset=dataset)
    deleted_cached_responses = delete_dataset_responses(dataset=dataset)
    if deleted_cached_responses is not None and deleted_cached_responses > 0:
        # delete assets
        cached_assets_storage_client.delete_dataset_directory(dataset)
        assets_storage_client.delete_dataset_directory(dataset)
    # create the jobs to backfill the dataset
    backfill_dataset(
        dataset=dataset,
        revision=revision,
        processing_graph=processing_graph,
        priority=priority,
        cache_max_days=1,
        blocked_datasets=blocked_datasets,
    )
    return RecreateDatasetReport(
        status="ok",
        dataset=dataset,
        cancelled_jobs=cancelled_jobs,
        deleted_cached_responses=deleted_cached_responses or 0,
    )


def create_recreate_dataset_endpoint(
    processing_graph: ProcessingGraph,
    cached_assets_storage_client: StorageClient,
    assets_storage_client: StorageClient,
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
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
                    cached_assets_storage_client=cached_assets_storage_client,
                    assets_storage_client=assets_storage_client,
                    blocked_datasets=blocked_datasets,
                    hf_endpoint=hf_endpoint,
                    hf_token=hf_token,
                    processing_graph=processing_graph,
                ),
                max_age=0,
            )
        except CustomError as e:
            return get_json_api_error_response(e, max_age=0)
        except Exception as e:
            return get_json_api_error_response(UnexpectedApiError("Unexpected error.", e), max_age=0)

    return recreate_dataset_endpoint
