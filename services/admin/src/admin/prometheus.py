# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Dict, Optional

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


class Prometheus:
    first_rows_queue: Queue
    metrics: Dict[str, Gauge] = {}
    split_queue: Queue
    prometheus_multiproc_dir: Optional[str]

    def __init__(self, prometheus_multiproc_dir: Optional[str]):
        self.prometheus_multiproc_dir = prometheus_multiproc_dir
        self.initMetrics()
        self.split_queue = Queue(type=JobType.SPLITS.value)
        self.first_rows_queue = Queue(type=JobType.FIRST_ROWS.value)

    def getRegistry(self) -> CollectorRegistry:
        # taken from https://github.com/perdy/starlette-prometheus/blob/master/starlette_prometheus/view.py
        if self.prometheus_multiproc_dir is not None:
            registry = CollectorRegistry()
            MultiProcessCollector(registry=registry, path=self.prometheus_multiproc_dir)
        else:
            registry = REGISTRY
        return registry

    # add metrics from the databases
    def initMetrics(self):
        self.metrics["queue_jobs_total"] = Gauge(
            "queue_jobs_total", "Number of jobs in the queue", ["queue", "status"]
        )
        self.metrics["cache_entries_total"] = Gauge(
            "cache_entries_total", "Number of entries in the cache", ["cache", "status"]
        )
        self.metrics["responses_in_cache_total"] = Gauge(
            "responses_in_cache_total",
            "Number of cached responses in the cache",
            ["path", "http_status", "error_code"],
        )

    def updateMetrics(self):
        # Queue metrics
        for status, total in self.split_queue.get_jobs_count_by_status().items():
            self.metrics["queue_jobs_total"].labels(queue=JobType.SPLITS.value, status=status).set(total)
        for status, total in self.first_rows_queue.get_jobs_count_by_status().items():
            self.metrics["queue_jobs_total"].labels(queue=JobType.FIRST_ROWS.value, status=status).set(total)
        # Cache metrics
        for http_status, by_error_code in get_splits_responses_count_by_status_and_error_code().items():
            for error_code, total in by_error_code.items():
                self.metrics["responses_in_cache_total"].labels(
                    path="/splits", http_status=http_status, error_code=error_code
                ).set(total)
        for http_status, by_error_code in get_first_rows_responses_count_by_status_and_error_code().items():
            for error_code, total in by_error_code.items():
                self.metrics["responses_in_cache_total"].labels(
                    path="/first-rows", http_status=http_status, error_code=error_code
                ).set(total)

    def endpoint(self, request: Request) -> Response:
        self.updateMetrics()

        return Response(generate_latest(self.getRegistry()), headers={"Content-Type": CONTENT_TYPE_LATEST})
