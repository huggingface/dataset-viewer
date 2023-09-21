# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from typing import Optional, TypedDict

from libcommon.dataset import get_supported_dataset_infos
from libcommon.simple_cache import (
    delete_dataset_responses,
    get_all_datasets,
    get_cache_count_for_dataset,
)
from libcommon.storage import StrPath
from libcommon.viewer_utils.asset import delete_asset_dir
from starlette.requests import Request
from starlette.responses import Response

from admin.authentication import auth_check
from admin.utils import (
    Endpoint,
    UnexpectedError,
    get_json_admin_error_response,
    get_json_ok_response,
)

MINIMUM_SUPPORTED_DATASETS = 20_000


class DatasetCacheReport(TypedDict):
    dataset: str
    cache_records: Optional[int]


def get_supported_dataset_names(
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> set[str]:
    supported_dataset_infos = get_supported_dataset_infos(hf_endpoint=hf_endpoint, hf_token=hf_token)
    return {dataset_info.id for dataset_info in supported_dataset_infos}


def get_obsolete_cache(
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> list[DatasetCacheReport]:
    supported_dataset_names = get_supported_dataset_names(hf_endpoint=hf_endpoint, hf_token=hf_token)
    existing_datasets = get_all_datasets()
    datasets_to_delete = existing_datasets.difference(supported_dataset_names)
    return [
        DatasetCacheReport(dataset=dataset, cache_records=get_cache_count_for_dataset(dataset=dataset))
        for dataset in datasets_to_delete
    ]


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
            auth_check(
                external_auth_url=external_auth_url,
                request=request,
                organization=organization,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            return get_json_ok_response(
                get_obsolete_cache(hf_endpoint=hf_endpoint, hf_token=hf_token), max_age=max_age
            )
        except Exception as e:
            return get_json_admin_error_response(UnexpectedError("Unexpected error.", e), max_age=max_age)

    return get_obsolete_cache_endpoint


def delete_obsolete_cache(
    hf_endpoint: str,
    assets_directory: StrPath,
    cached_assets_directory: StrPath,
    hf_token: Optional[str] = None,
) -> list[DatasetCacheReport]:
    supported_dataset_names = get_supported_dataset_names(hf_endpoint=hf_endpoint, hf_token=hf_token)
    if len(supported_dataset_names) < MINIMUM_SUPPORTED_DATASETS:
        raise UnexpectedError(f"only {len(supported_dataset_names)} datasets were found")

    existing_datasets = get_all_datasets()
    datasets_to_delete = existing_datasets.difference(supported_dataset_names)

    deletion_report = []
    for dataset in datasets_to_delete:
        # delete cache records
        datasets_cache_records = delete_dataset_responses(dataset=dataset)
        if datasets_cache_records is not None and datasets_cache_records > 0:
            # delete assets
            delete_asset_dir(dataset=dataset, directory=assets_directory)
            delete_asset_dir(dataset=dataset, directory=cached_assets_directory)
            logging.debug(f"{dataset} has been delete with {datasets_cache_records} cache records")
        else:
            logging.debug(f"unable to delete {dataset}")
        deletion_report.append(DatasetCacheReport(dataset=dataset, cache_records=datasets_cache_records))

    return deletion_report


def create_delete_obsolete_cache_endpoint(
    hf_endpoint: str,
    max_age: int,
    assets_directory: StrPath,
    cached_assets_directory: StrPath,
    hf_token: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Endpoint:
    async def delete_obsolete_cache_endpoint(request: Request) -> Response:
        try:
            logging.info("/obsolete-cache")
            auth_check(
                external_auth_url=external_auth_url,
                request=request,
                organization=organization,
                hf_timeout_seconds=hf_timeout_seconds,
            )

            return get_json_ok_response(
                delete_obsolete_cache(
                    hf_endpoint=hf_endpoint,
                    hf_token=hf_token,
                    assets_directory=assets_directory,
                    cached_assets_directory=cached_assets_directory,
                ),
                max_age=max_age,
            )
        except Exception as e:
            return get_json_admin_error_response(UnexpectedError("Unexpected error.", e), max_age=max_age)

    return delete_obsolete_cache_endpoint
