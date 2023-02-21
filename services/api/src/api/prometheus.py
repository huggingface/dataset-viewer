# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os
from typing import Any

from prometheus_client import (  # type: ignore
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    generate_latest,
)
from prometheus_client.multiprocess import MultiProcessCollector  # type: ignore

# ^ type: ignore can be removed on next release:
# https://github.com/prometheus/client_python/issues/491#issuecomment-1429287314
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

    def getLatestContent(self) -> Any:
        # ^ returns Any because we cannot be sure latest are UTF8Bytes
        latest = generate_latest(self.getRegistry())
        return latest.decode("utf-8")

    def endpoint(self, request: Request) -> Response:
        return Response(self.getLatestContent(), headers={"Content-Type": CONTENT_TYPE_LATEST})
