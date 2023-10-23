# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libapi.utils import Endpoint
from libcommon.prometheus import (
    Prometheus,
    update_descriptive_statistics_disk_usage,
    update_duckdb_disk_usage,
    update_hf_datasets_disk_usage,
    update_parquet_metadata_disk_usage,
    update_queue_jobs_total,
    update_responses_in_cache_total,
)
from libcommon.storage import StrPath
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.requests import Request
from starlette.responses import Response


def create_metrics_endpoint(
    descriptive_statistics_directory: StrPath,
    duckdb_directory: StrPath,
    hf_datasets_directory: StrPath,
    parquet_metadata_directory: StrPath,
) -> Endpoint:
    prometheus = Prometheus()

    async def metrics_endpoint(_: Request) -> Response:
        logging.info("/metrics")
        update_queue_jobs_total()
        update_responses_in_cache_total()
        # TODO: Update disk usage from fsspec
        update_descriptive_statistics_disk_usage(directory=descriptive_statistics_directory)
        update_duckdb_disk_usage(directory=duckdb_directory)
        update_hf_datasets_disk_usage(directory=hf_datasets_directory)
        update_parquet_metadata_disk_usage(directory=parquet_metadata_directory)
        return Response(prometheus.getLatestContent(), headers={"Content-Type": CONTENT_TYPE_LATEST})

    return metrics_endpoint
