# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os

from libcache.simple_cache import (
    get_first_rows_responses_count_by_status_and_error_code,
    get_splits_responses_count_by_status_and_error_code,
)
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
    labelnames=["path", "http_status", "error_code"],
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
        for http_status, by_error_code in get_splits_responses_count_by_status_and_error_code().items():
            for error_code, total in by_error_code.items():
                RESPONSES_IN_CACHE_TOTAL.labels(path="/splits", http_status=http_status, error_code=error_code).set(
                    total
                )
        for http_status, by_error_code in get_first_rows_responses_count_by_status_and_error_code().items():
            for error_code, total in by_error_code.items():
                RESPONSES_IN_CACHE_TOTAL.labels(
                    path="/first-rows", http_status=http_status, error_code=error_code
                ).set(total)

    def getLatestContent(self) -> str:
        self.updateMetrics()
        return generate_latest(self.getRegistry()).decode("utf-8")

    def endpoint(self, request: Request) -> Response:
        return Response(self.getLatestContent(), headers={"Content-Type": CONTENT_TYPE_LATEST})
