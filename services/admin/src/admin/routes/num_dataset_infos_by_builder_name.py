# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.simple_cache import CachedResponseDocument
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


def create_num_dataset_infos_by_builder_name_endpoint(
    max_age: int,
    external_auth_url: Optional[str] = None,
    organization: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Endpoint:
    async def usage_endpoint(request: Request) -> Response:
        try:
            logging.info("/num-dataset-infos-by-builder-name")

            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(
                external_auth_url=external_auth_url,
                request=request,
                organization=organization,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            num_datasets_infos = CachedResponseDocument.objects(http_status=200, kind="dataset-info").count()
            num_parquet_datasets_infos = CachedResponseDocument.objects(
                http_status=200, kind="dataset-info", content__dataset_info__default__builder_name="parquet"
            ).count()
            num_csv_datasets_infos = CachedResponseDocument.objects(
                http_status=200, kind="dataset-info", content__dataset_info__default__builder_name="csv"
            ).count()
            num_text_datasets_infos = CachedResponseDocument.objects(
                http_status=200, kind="dataset-info", content__dataset_info__default__builder_name="text"
            ).count()
            num_imagefolder_datasets_infos = CachedResponseDocument.objects(
                http_status=200, kind="dataset-info", content__dataset_info__default__builder_name="imagefolder"
            ).count()
            num_audiofolder_datasets_infos = CachedResponseDocument.objects(
                http_status=200, kind="dataset-info", content__dataset_info__default__builder_name="audiofolder"
            ).count()
            num_json_datasets_infos = CachedResponseDocument.objects(
                http_status=200, kind="dataset-info", content__dataset_info__default__builder_name="json"
            ).count()
            num_arrow_datasets_infos = CachedResponseDocument.objects(
                http_status=200, kind="dataset-info", content__dataset_info__default__builder_name="arrow"
            ).count()
            num_other_dataset_infos = num_datasets_infos - (
                num_parquet_datasets_infos
                + num_csv_datasets_infos
                + num_text_datasets_infos
                + num_imagefolder_datasets_infos
                + num_audiofolder_datasets_infos
                + num_json_datasets_infos
            )
            return get_json_ok_response(
                {
                    "parquet": num_parquet_datasets_infos,
                    "csv": num_csv_datasets_infos,
                    "text": num_text_datasets_infos,
                    "imagefolder": num_imagefolder_datasets_infos,
                    "audiofolder": num_audiofolder_datasets_infos,
                    "json": num_json_datasets_infos,
                    "arrow": num_arrow_datasets_infos,
                    "other": num_other_dataset_infos,
                },
                max_age=max_age,
            )
        except AdminCustomError as e:
            return get_json_admin_error_response(e, max_age=max_age)
        except Exception as e:
            return get_json_admin_error_response(UnexpectedError("Unexpected error.", e), max_age=max_age)

    return usage_endpoint
