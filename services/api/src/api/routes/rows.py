# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, Optional

import hffs
import pyarrow.parquet as pq
from libcommon.processing_graph import ProcessingStep
from libcommon.simple_cache import DoesNotExist, get_response
from starlette.requests import Request
from starlette.responses import Response

from api.authentication import auth_check
from api.utils import (
    ApiCustomError,
    Endpoint,
    MissingRequiredParameterError,
    ResponseNotFoundError,
    UnexpectedError,
    are_valid_parameters,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
)


def create_rows_endpoint(
    parquet_processing_step: ProcessingStep,
    hf_jwt_public_key: Optional[Any] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Endpoint:
    async def rows_endpoint(request: Request) -> Response:
        try:
            dataset = request.query_params.get("dataset")
            config = request.query_params.get("config")
            split = request.query_params.get("split")
            if not dataset or not are_valid_parameters([dataset, config, split]):
                raise MissingRequiredParameterError("Parameter 'dataset', 'config' and 'split' are required")
            MAX_ROWS = 100
            fromRow = int(request.query_params.get("from", 0))
            _toRow = request.query_params.get("to")
            if _toRow is None:
                toRow = fromRow + MAX_ROWS
            else:
                toRow = max(min(int(_toRow), fromRow + MAX_ROWS), fromRow)
            logging.info("/rows, dataset={dataset}, config={config}, split={split}, from={fromRow}, to={toRow}}")

            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(
                dataset,
                external_auth_url=external_auth_url,
                request=request,
                hf_jwt_public_key=hf_jwt_public_key,
                hf_jwt_algorithm=hf_jwt_algorithm,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            try:
                # get the list of parquet files
                result = get_response(kind=parquet_processing_step.cache_kind, dataset=dataset)
                content = result["content"]
                http_status = result["http_status"]
                error_code = result["error_code"]
                if http_status != HTTPStatus.OK:
                    # TODO: send a proper error message, saying that the parquet files cannot be created
                    return get_json_error_response(
                        content=content, status_code=http_status, max_age=max_age_short, error_code=error_code
                    )
                parquet_files_paths = [
                    f"{config}/{parquet_file['filename']}"
                    for parquet_file in content["parquet_files"]
                    if parquet_file["split"] == split and parquet_file["config"] == config
                ]
                # get the list of the first 100 rows
                REVISION = "refs/convert/parquet"
                fs = hffs.HfFileSystem(dataset, repo_type="dataset", revision=REVISION)
                parquet_dataset = pq.ParquetDataset(parquet_files_paths, filesystem=fs)
                table = parquet_dataset.read()
                rows = table.take(list(range(fromRow, toRow))).to_pylist()
                return get_json_ok_response(content={"rows": rows}, max_age=max_age_long)
            except DoesNotExist as e:
                # add "check_in_process" ...
                raise ResponseNotFoundError("Not found.") from e
        except ApiCustomError as e:
            return get_json_api_error_response(error=e, max_age=max_age_short)
        except Exception as e:
            return get_json_api_error_response(error=UnexpectedError("Unexpected error.", e), max_age=max_age_short)

    return rows_endpoint
