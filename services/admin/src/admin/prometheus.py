# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os

from libcache.simple_cache import get_responses_count_by_kind_status_and_error_code
from libqueue.queue import Queue
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

from admin.utils import JobType

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
    first_rows_queue: Queue
    split_queue: Queue

    def __init__(self):
        self.split_queue = Queue(type=JobType.SPLITS.value)
        self.first_rows_queue = Queue(type=JobType.FIRST_ROWS.value)

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
        for status, total in self.split_queue.get_jobs_count_by_status().items():
            QUEUE_JOBS_TOTAL.labels(queue=JobType.SPLITS.value, status=status).set(total)
        for status, total in self.first_rows_queue.get_jobs_count_by_status().items():
            QUEUE_JOBS_TOTAL.labels(queue=JobType.FIRST_ROWS.value, status=status).set(total)
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
