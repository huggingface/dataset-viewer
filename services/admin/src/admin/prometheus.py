# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os

from libcommon.processing_steps import PROCESSING_STEPS
from libcommon.queue import Queue
from libcommon.simple_cache import get_responses_count_by_kind_status_and_error_code
from prometheus_client import (  # type: ignore # https://github.com/prometheus/client_python/issues/491
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Gauge,
    generate_latest,
)
from prometheus_client.multiprocess import (  # type: ignore # https://github.com/prometheus/client_python/issues/491
    MultiProcessCollector,
)
from starlette.requests import Request
from starlette.responses import Response

# the metrics are global to the process
QUEUE_JOBS_TOTAL = Gauge(
    name="queue_jobs_total",
    documentation="Number of jobs in the queue",
    labelnames=["queue", "status"],
    multiprocess_mode="liveall",
)
RESPONSES_IN_CACHE_TOTAL = Gauge(
    name="responses_in_cache_total",
    documentation="Number of cached responses in the cache",
    labelnames=["kind", "http_status", "error_code"],
    multiprocess_mode="liveall",
)


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

    def updateMetrics(self):
        # Queue metrics
        for processing_step in PROCESSING_STEPS:
            for status, total in Queue(type=processing_step.job_type).get_jobs_count_by_status().items():
                QUEUE_JOBS_TOTAL.labels(queue=processing_step.job_type, status=status).set(total)
        # Cache metrics
        for metric in get_responses_count_by_kind_status_and_error_code():
            RESPONSES_IN_CACHE_TOTAL.labels(
                kind=metric["kind"], http_status=metric["http_status"], error_code=metric["error_code"]
            ).set(metric["count"])

    def getLatestContent(self) -> str:
        self.updateMetrics()
        return generate_latest(self.getRegistry()).decode("utf-8")

    def endpoint(self, request: Request) -> Response:
        return Response(self.getLatestContent(), headers={"Content-Type": CONTENT_TYPE_LATEST})
