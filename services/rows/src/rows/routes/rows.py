# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from functools import lru_cache
from http import HTTPStatus
from typing import Optional

from fsspec.implementations.http import HTTPFileSystem
from libapi.authentication import auth_check, get_auth_kind
from libapi.exceptions import (
    ApiError,
    ResponseNotFoundError,
    TooBigContentError,
    UnexpectedApiError,
)
from libapi.request import (
    get_caller_debug_info,
    get_request_parameter,
    get_request_parameter_length,
    get_request_parameter_offset,
    get_request_source,
)
from libapi.response import create_response
from libapi.utils import (
    Endpoint,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
    try_backfill_dataset_then_raise,
)
from libcommon.constants import CONFIG_PARQUET_METADATA_KIND
from libcommon.parquet_utils import EmptyParquetMetadataError, RowsIndex, TooBigRows
from libcommon.prometheus import StepProfiler
from libcommon.simple_cache import CachedArtifactError, CachedArtifactNotFoundError
from libcommon.storage import StrPath
from libcommon.storage_client import StorageClient
from starlette.requests import Request
from starlette.responses import Response

from rows.arrow_budget import ArrowBudget
from rows.memory import record_arrow_bytes, track_request

logger = logging.getLogger(__name__)

# the user-agent is free-form (and spoofable): we cap it to keep the log lines bounded
MAX_USER_AGENT_LENGTH = 200
# the datasets whose owner asked to log the address of every caller, to debug how the calls reach the service (see
# get_caller_debug_info). An address is a personal data: only list a dataset here while its owner needs it.
PROBE_DATASETS = {"lhoestq/probe-rows-service"}
# a request that waits longer for a query slot gives up before the load balancer does, and tells the client to retry
QUERY_SLOT_WAIT_SECONDS = 20


def create_rows_endpoint(
    cached_assets_storage_client: StorageClient,
    parquet_metadata_directory: StrPath,
    max_arrow_data_in_memory: int,
    max_concurrent_queries: int,
    max_arrow_data_per_worker: int,
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    hf_jwt_public_keys: Optional[list[str]] = None,
    hf_jwt_algorithm: Optional[str] = None,
    hf_jwt_issuers: Optional[list[str]] = None,
    external_auth_url: Optional[str] = None,
    hub_origins: Optional[list[str]] = None,
    hub_networks: Optional[list[str]] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
    storage_clients: Optional[list[StorageClient]] = None,
) -> Endpoint:
    httpfs = HTTPFileSystem(headers={"authorization": f"Bearer {hf_token}"})
    # Each query holds its Arrow table until its response is built, and the event loop starts every request it
    # receives: without a bound, the queries at once outgrow the pod memory. A query slot is a share of the budget,
    # so that a worker runs at most max_concurrent_queries queries at once, fewer when their tables are big.
    arrow_budget = ArrowBudget(max_arrow_data_per_worker)
    query_slot_bytes = max_arrow_data_per_worker // max_concurrent_queries

    @lru_cache(maxsize=1)
    def get_rows_index(dataset: str, config: str, split: str) -> RowsIndex:
        # cache the RowsIndex instance and therefore save one call to Mongo
        # if multiple queries to the same dataset are done in a row (90% of
        # requests in a short time window are to the same dataset)
        return RowsIndex(
            dataset=dataset,
            config=config,
            split=split,
            hf_token=hf_token,
            hf_endpoint=hf_endpoint,
            max_scan_size=max_arrow_data_in_memory,
            parquet_metadata_directory=parquet_metadata_directory,
        )

    async def rows_endpoint(request: Request) -> Response:
        await httpfs.set_session()
        revision: Optional[str] = None
        with StepProfiler(method="rows_endpoint", step="all"):
            try:
                with StepProfiler(method="rows_endpoint", step="validate parameters"):
                    dataset = get_request_parameter(request, "dataset", required=True)
                    config = get_request_parameter(request, "config", required=True)
                    split = get_request_parameter(request, "split", required=True)
                    offset = get_request_parameter_offset(request)
                    length = get_request_parameter_length(request)
                    # "source" and "auth_kind" tell apart the calls coming from the Hugging Face Hub (its servers,
                    # or the browser on huggingface.co) from the calls of the external users. They are read from the
                    # request, so they are logging labels, not an authentication (see get_request_source).
                    source = get_request_source(request, hub_origins=hub_origins, hub_networks=hub_networks)
                    auth_kind = get_auth_kind(request)
                    origin = request.headers.get("origin", "-")
                    user_agent = request.headers.get("user-agent", "-")[:MAX_USER_AGENT_LENGTH]
                    logging.info(
                        f"/rows, {dataset=}, {config=}, {split=}, {offset=}, {length=}, "
                        f"{source=}, {auth_kind=}, {origin=}, {user_agent=}"
                    )
                    if dataset in PROBE_DATASETS:
                        details = get_caller_debug_info(request)
                        # the configured networks are logged as well: a "hub-server" label is only reachable if the
                        # caller address falls in one of them, so they tell if the configuration reached the service
                        logging.info(f"/rows caller, {dataset=}, {source=}, {hub_networks=}, {details=}")
                if dataset == "HuggingFaceFW/fineweb-edu-score-2" and offset > 1_000_000:
                    return get_json_error_response(
                        content={"error": "too many requests"},
                        status_code=HTTPStatus.TOO_MANY_REQUESTS,
                        max_age=max_age_short,
                    )
                with StepProfiler(method="rows_endpoint", step="check authentication"):
                    # if auth_check fails, it will raise an exception that will be caught below
                    await auth_check(
                        dataset=dataset,
                        external_auth_url=external_auth_url,
                        request=request,
                        hf_jwt_public_keys=hf_jwt_public_keys,
                        hf_jwt_algorithm=hf_jwt_algorithm,
                        hf_jwt_issuers=hf_jwt_issuers,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )
                try:
                    with StepProfiler(method="rows_endpoint", step="get row groups index"):
                        rows_index = get_rows_index(dataset=dataset, config=config, split=split)
                        revision = rows_index.revision
                    with StepProfiler(method="rows_endpoint", step="get the size of the rows"):
                        scan_size = await rows_index.scan_size(offset=offset, length=length)
                    query_bytes = max(min(scan_size, max_arrow_data_in_memory), query_slot_bytes)
                    async with arrow_budget.hold(query_bytes, wait_seconds=QUERY_SLOT_WAIT_SECONDS) as hold_at_least:
                        with track_request(f"{dataset=} {config=} {split=} {offset=} {length=}") as tracked:
                            with StepProfiler(method="rows_endpoint", step="query the rows"):
                                try:
                                    # Some datasets have very long binary data that we truncate
                                    pa_table, truncated_columns = await rows_index.query(offset=offset, length=length)
                                except TooBigRows as err:
                                    raise TooBigContentError(str(err)) from None
                            record_arrow_bytes(tracked, pa_table.nbytes)
                            hold_at_least(pa_table.nbytes)
                            with StepProfiler(method="rows_endpoint", step="transform to a list"):
                                response = await create_response(
                                    dataset=dataset,
                                    revision=revision,
                                    config=config,
                                    split=split,
                                    storage_client=cached_assets_storage_client,
                                    hf_endpoint=hf_endpoint,
                                    hf_token=hf_token,
                                    pa_table=pa_table,
                                    offset=offset,
                                    features=rows_index.features,
                                    partial=rows_index.partial,
                                    num_rows_total=rows_index.num_rows_total,
                                    truncated_columns=truncated_columns,
                                )
                except EmptyParquetMetadataError:
                    raise ResponseNotFoundError("Not found.") from None
                except CachedArtifactNotFoundError:
                    with StepProfiler(method="rows_endpoint", step="try backfill dataset"):
                        try_backfill_dataset_then_raise(
                            processing_step_name=CONFIG_PARQUET_METADATA_KIND,
                            dataset=dataset,
                            hf_endpoint=hf_endpoint,
                            hf_timeout_seconds=hf_timeout_seconds,
                            hf_token=hf_token,
                            blocked_datasets=blocked_datasets,
                            storage_clients=storage_clients,
                        )
                with StepProfiler(method="rows_endpoint", step="generate the OK response"):
                    return get_json_ok_response(content=response, max_age=max_age_long, revision=revision)
            except CachedArtifactError as e:
                content = e.cache_entry_with_details["content"]
                http_status = e.cache_entry_with_details["http_status"]
                error_code = e.cache_entry_with_details["error_code"]
                return get_json_error_response(
                    content=content,
                    status_code=http_status,
                    max_age=max_age_short,
                    error_code=error_code,
                    revision=revision,
                )
            except Exception as e:
                error = e if isinstance(e, ApiError) else UnexpectedApiError("Unexpected error.", e)
                # the scan size only depends on the parquet files, so a retry gets the same error
                max_age = max_age_long if isinstance(error, TooBigContentError) else max_age_short
                with StepProfiler(method="rows_endpoint", step="generate API error response"):
                    return get_json_api_error_response(error=error, max_age=max_age, revision=revision)

    return rows_endpoint
