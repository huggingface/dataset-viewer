# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

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
    prometheus_multiproc_dir: Optional[str]

    def __init__(self, prometheus_multiproc_dir: Optional[str]):
        self.prometheus_multiproc_dir = prometheus_multiproc_dir

    def getRegistry(self) -> CollectorRegistry:
        # taken from https://github.com/perdy/starlette-prometheus/blob/master/starlette_prometheus/view.py
        if self.prometheus_multiproc_dir is not None:
            registry = CollectorRegistry()
            MultiProcessCollector(registry=registry, path=self.prometheus_multiproc_dir)
        else:
            registry = REGISTRY
        return registry

    def endpoint(self, request: Request) -> Response:
        return Response(generate_latest(self.getRegistry()), headers={"Content-Type": CONTENT_TYPE_LATEST})
