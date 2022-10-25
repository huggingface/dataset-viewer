# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os

from prometheus_client import (  # type: ignore # https://github.com/prometheus/client_python/issues/491
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    generate_latest,
)
from prometheus_client.multiprocess import (  # type: ignore # https://github.com/prometheus/client_python/issues/491
    MultiProcessCollector,
)
from starlette.requests import Request
from starlette.responses import Response


class Prometheus:
    def getRegistry(self) -> CollectorRegistry:
        # taken from https://github.com/perdy/starlette-prometheus/blob/master/starlette_prometheus/view.py
        # see https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn
        if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
            registry = CollectorRegistry()
            MultiProcessCollector(registry=registry)
        else:
            registry = REGISTRY
        return registry

    def getLatestContent(self) -> str:
        return generate_latest(self.getRegistry()).decode("utf-8")

    def endpoint(self, request: Request) -> Response:
        return Response(self.getLatestContent(), headers={"Content-Type": CONTENT_TYPE_LATEST})
