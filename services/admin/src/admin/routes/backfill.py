# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.dataset import get_supported_datasets
from libcommon.operations import update_dataset
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
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


def create_backfill_endpoint(
    init_processing_steps: list[ProcessingStep],
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
) -> Endpoint:
    async def backfill_endpoint(request: Request) -> Response:
        try:
            logging.info("/backfill")

            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(external_auth_url=external_auth_url, request=request, organization=organization)
            for dataset in get_supported_datasets(hf_endpoint=hf_endpoint, hf_token=hf_token):
                update_dataset(
                    dataset=dataset,
                    init_processing_steps=init_processing_steps,
                    hf_endpoint=hf_endpoint,
                    hf_token=hf_token,
                    force=False,
                    priority=Priority.LOW,
                    do_check_support=False,
                )
            # ^ we simply ask an update for all the datasets on the Hub, supported by the datasets-server
            # we could be more precise and only ask for updates for the datasets that have some missing
            # cache entries, but it's not easy to check.
            # Also: we could try to do a batch update of the database, instead of one query per dataset
            return get_json_ok_response(
                {"status": "ok"},
                max_age=0,
            )
        except AdminCustomError as e:
            return get_json_admin_error_response(e, max_age=0)
        except Exception as e:
            return get_json_admin_error_response(UnexpectedError("Unexpected error.", e), max_age=0)

    return backfill_endpoint
