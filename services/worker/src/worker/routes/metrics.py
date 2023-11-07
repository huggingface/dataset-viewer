# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from libcommon.prometheus import (
    Prometheus,
)
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.requests import Request
from starlette.responses import Response

Endpoint = Callable[[Request], Coroutine[Any, Any, Response]]


def create_metrics_endpoint() -> Endpoint:
    prometheus = Prometheus()

    async def metrics_endpoint(_: Request) -> Response:
        logging.info("/metrics")
        return Response(prometheus.getLatestContent(), headers={"Content-Type": CONTENT_TYPE_LATEST})

    return metrics_endpoint
