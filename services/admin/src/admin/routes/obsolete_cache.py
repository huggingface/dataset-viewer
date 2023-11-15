# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from typing import Optional

from libapi.exceptions import UnexpectedApiError
from libapi.utils import Endpoint, get_json_api_error_response, get_json_ok_response
from libcommon.obsolete_cache import delete_obsolete_cache, get_obsolete_cache
from libcommon.storage_client import StorageClient
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check


def create_get_obsolete_cache_endpoint(
    hf_endpoint: str,
    max_age: int,
    hf_token: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Endpoint:
    async def get_obsolete_cache_endpoint(request: Request) -> Response:
        try:
            logging.info("/obsolete-cache")
            await auth_check(
                external_auth_url=external_auth_url,
                request=request,
                organization=organization,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            return get_json_ok_response(
                get_obsolete_cache(hf_endpoint=hf_endpoint, hf_token=hf_token), max_age=max_age
            )
        except Exception as e:
            return get_json_api_error_response(UnexpectedApiError("Unexpected error.", e), max_age=max_age)

    return get_obsolete_cache_endpoint


def create_delete_obsolete_cache_endpoint(
    hf_endpoint: str,
    max_age: int,
    cached_assets_storage_client: StorageClient,
    assets_storage_client: StorageClient,
    hf_token: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Endpoint:
    async def delete_obsolete_cache_endpoint(request: Request) -> Response:
        try:
            logging.info("/obsolete-cache")
            await auth_check(
                external_auth_url=external_auth_url,
                request=request,
                organization=organization,
                hf_timeout_seconds=hf_timeout_seconds,
            )

            return get_json_ok_response(
                delete_obsolete_cache(
                    hf_endpoint=hf_endpoint,
                    hf_token=hf_token,
                    cached_assets_storage_client=cached_assets_storage_client,
                    assets_storage_client=assets_storage_client,
                ),
                max_age=max_age,
            )
        except Exception as e:
            return get_json_api_error_response(UnexpectedApiError("Unexpected error.", e), max_age=max_age)

    return delete_obsolete_cache_endpoint
