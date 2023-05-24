# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.prometheus import (
    Prometheus,
    update_assets_disk_usage,
    update_queue_jobs_total,
    update_responses_in_cache_total,
)
from libcommon.storage import StrPath
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.requests import Request
from starlette.responses import Response

from admin.utils import Endpoint


def create_metrics_endpoint(assets_directory: StrPath) -> Endpoint:
    prometheus = Prometheus()

    async def metrics_endpoint(_: Request) -> Response:
        logging.info("/metrics")
        update_queue_jobs_total()
        update_responses_in_cache_total()
        update_assets_disk_usage(assets_directory=assets_directory)
        return Response(prometheus.getLatestContent(), headers={"Content-Type": CONTENT_TYPE_LATEST})

    return metrics_endpoint
