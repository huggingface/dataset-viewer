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
        if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
            registry = CollectorRegistry()
            MultiProcessCollector(registry)
        else:
            registry = REGISTRY
        return registry

    def endpoint(self, request: Request) -> Response:
        return Response(generate_latest(self.getRegistry()), headers={"Content-Type": CONTENT_TYPE_LATEST})
