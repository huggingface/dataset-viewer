# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os
import time
from types import TracebackType
from typing import Any, Optional, Type, TypeVar

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.multiprocess import MultiProcessCollector
from psutil import disk_usage

from libcommon.metrics import JobTotalMetricDocument
from libcommon.simple_cache import CacheTotalMetricDocument
from libcommon.storage import StrPath


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
METHOD_STEPS_PROCESSING_TIME = Histogram(
    "method_steps_processing_time_seconds",
    "Histogram of the processing time of specific steps in methods for a given context (in seconds)",
    ["method", "step", "context"],
)


def update_queue_jobs_total() -> None:
    for job_metric in JobTotalMetricDocument.objects():
        QUEUE_JOBS_TOTAL.labels(queue=job_metric.queue, status=job_metric.status).set(job_metric.total)


def update_responses_in_cache_total() -> None:
    for cache_metric in CacheTotalMetricDocument.objects():
        RESPONSES_IN_CACHE_TOTAL.labels(
            kind=cache_metric.kind, http_status=cache_metric.http_status, error_code=cache_metric.error_code
        ).set(cache_metric.total)


def update_assets_disk_usage(assets_directory: StrPath) -> None:
    # TODO: move to metrics, as for the other metrics (queue, cache)
    total, used, free, percent = disk_usage(str(assets_directory))
    ASSETS_DISK_USAGE.labels(type="total").set(total)
    ASSETS_DISK_USAGE.labels(type="used").set(used)
    ASSETS_DISK_USAGE.labels(type="free").set(free)
    ASSETS_DISK_USAGE.labels(type="percent").set(percent)


T = TypeVar("T", bound="StepProfiler")


class StepProfiler:
    """
    A context manager that measures the time spent in a step of a method and reports it to Prometheus.

    Example:
        >>> with StepProfiler("method", "step", "context") as profiler:
        ...     pass

    Args:
        method (str): The name of the method.
        step (str): The name of the step.
        context (str|None): An optional string that adds context. If None, the label "None" is used.
    """

    def __init__(self, method: str, step: str, context: Optional[str] = None):
        self.method = method
        self.step = step
        self.context = str(context)
        self.before_time = time.perf_counter()

    def __enter__(self: T) -> T:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        after_time = time.perf_counter()
        METHOD_STEPS_PROCESSING_TIME.labels(method=self.method, step=self.step, context=self.context).observe(
            after_time - self.before_time
        )
