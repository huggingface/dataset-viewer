# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from typing import Optional

from libapi.exceptions import InvalidParameterError, UnexpectedApiError
from libapi.request import get_required_request_parameter
from libapi.utils import Endpoint, get_json_api_error_response, get_json_ok_response
from libcommon.dataset import get_dataset_git_revision
from libcommon.exceptions import CustomError
from libcommon.operations import backfill_dataset
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue
from libcommon.simple_cache import delete_dataset_responses
from libcommon.storage import StrPath
from libcommon.utils import Priority
from libcommon.viewer_utils.asset import delete_asset_dir
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check


def create_recreate_dataset_endpoint(
    processing_graph: ProcessingGraph,
    assets_directory: StrPath,
    cached_assets_directory: StrPath,
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Endpoint:
    async def recreate_dataset_endpoint(request: Request) -> Response:
        try:
            dataset = get_required_request_parameter(request, "dataset")
            try:
                priority = Priority(request.query_params.get("priority", "low"))
            except ValueError:
                raise InvalidParameterError(
                    f"Parameter 'priority' should be one of {', '.join(prio.value for prio in Priority)}."
                )
            logging.info(f"/recreate-dataset, dataset={dataset}, priority={priority}")

            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(
                external_auth_url=external_auth_url,
                request=request,
                organization=organization,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            # try to get the revision of the dataset (before deleting the jobs and the cache, in case it fails)
            revision = get_dataset_git_revision(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token)
            # delete all the jobs and all the cache entries for the dataset (for all the revisions)
            cancelled_jobs = Queue().cancel_dataset_jobs(dataset=dataset)
            deleted_cached_responses = delete_dataset_responses(dataset=dataset)
            if deleted_cached_responses is not None and deleted_cached_responses > 0:
                # delete assets
                delete_asset_dir(dataset=dataset, directory=assets_directory)
                delete_asset_dir(dataset=dataset, directory=cached_assets_directory)
            # create the jobs to backfill the dataset
            backfill_dataset(
                dataset=dataset,
                revision=revision,
                processing_graph=processing_graph,
                priority=priority,
                cache_max_days=1,
                blocked_datasets=blocked_datasets,
            )
            return get_json_ok_response(
                {
                    "status": "ok",
                    "dataset": dataset,
                    "cancelled_jobs": cancelled_jobs,
                    "deleted_cached_responses": deleted_cached_responses or 0,
                },
                max_age=0,
            )
        except CustomError as e:
            return get_json_api_error_response(e, max_age=0)
        except Exception as e:
            return get_json_api_error_response(UnexpectedApiError("Unexpected error.", e), max_age=0)

    return recreate_dataset_endpoint
