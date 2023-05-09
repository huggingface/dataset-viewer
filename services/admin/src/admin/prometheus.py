# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os
from dataclasses import dataclass
from typing import Any

from libcommon.metrics import CacheTotalMetric, JobTotalMetric
from libcommon.processing_graph import ProcessingGraph
from libcommon.utils import Status
from libcommon.storage import StrPath
from libcommon.utils import Status
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Gauge,
    generate_latest,
)
from prometheus_client.multiprocess import MultiProcessCollector

# ^ type: ignore can be removed on next release:
# https://github.com/prometheus/client_python/issues/491#issuecomment-1429287314
from psutil import disk_usage
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
ASSETS_DISK_USAGE = Gauge(
    name="assets_disk_usage",
    documentation="Usage of the disk where the assets are stored",
    labelnames=["type"],
    multiprocess_mode="liveall",
)


@dataclass
class Prometheus:
    processing_graph: ProcessingGraph
    assets_directory: StrPath

    def getRegistry(self) -> CollectorRegistry:
        # taken from https://github.com/perdy/starlette-prometheus/blob/master/starlette_prometheus/view.py
        # see https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn
        if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
            registry = CollectorRegistry()
            MultiProcessCollector(registry=registry)
        else:
            registry = REGISTRY
        return registry

    def updateMetrics(self) -> None:
        # Queue metrics
        if queue_jobs_total := JobTotalMetric.objects():
            for job_metric in queue_jobs_total:
                QUEUE_JOBS_TOTAL.labels(queue=job_metric.queue, status=job_metric.status).set(job_metric.total)
        else:
            # TODO: Move this logic to a metrics manager
            # In case job collected metrics do not exist, fill with 0
            for processing_step in self.processing_graph.get_processing_steps():
                for status in Status:
                    QUEUE_JOBS_TOTAL.labels(queue=processing_step.job_type, status=status.value).set(0)

        # Cache metrics
        if responses_in_cache_total := CacheTotalMetric.objects():
            for cache_metric in responses_in_cache_total:
                RESPONSES_IN_CACHE_TOTAL.labels(
                    kind=cache_metric.kind, http_status=cache_metric.http_status, error_code=cache_metric.error_code
                ).set(cache_metric.total)
        else:
            # TODO: Move this logic to a metrics manager
            # In case cache collected metrics do not exist, fill with 0
            for processing_step in self.processing_graph.get_processing_steps():
                RESPONSES_IN_CACHE_TOTAL.labels(
                    kind=processing_step.cache_kind, http_status="200", error_code="None"
                ).set(0)

        # Assets storage metrics
        total, used, free, percent = disk_usage(str(self.assets_directory))
        ASSETS_DISK_USAGE.labels(type="total").set(total)
        ASSETS_DISK_USAGE.labels(type="used").set(used)
        ASSETS_DISK_USAGE.labels(type="free").set(free)
        ASSETS_DISK_USAGE.labels(type="percent").set(percent)

    def getLatestContent(self) -> Any:
        # ^ returns Any because we cannot be sure latest are UTF8Bytes
        self.updateMetrics()
        latest = generate_latest(self.getRegistry())
        return latest.decode("utf-8")

    def endpoint(self, request: Request) -> Response:
        return Response(self.getLatestContent(), headers={"Content-Type": CONTENT_TYPE_LATEST})
