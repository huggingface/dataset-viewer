# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os
import time
from types import TracebackType
from typing import Any, Optional, TypeVar

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.multiprocess import MultiProcessCollector
from psutil import disk_usage

from libcommon.constants import LONG_DURATION_PROMETHEUS_HISTOGRAM_BUCKETS
from libcommon.queue.metrics import JobTotalMetricDocument, WorkerSizeJobsCountDocument
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
    labelnames=["queue", "status", "dataset_status"],
    multiprocess_mode="liveall",
)
WORKER_SIZE_JOBS_COUNT = Gauge(
    name="worker_size_jobs_count",
    documentation="Number of jobs per worker size",
    labelnames=["worker_size"],
    multiprocess_mode="liveall",
)
RESPONSES_IN_CACHE_TOTAL = Gauge(
    name="responses_in_cache_total",
    documentation="Number of cached responses in the cache",
    labelnames=["kind", "http_status", "error_code"],
    multiprocess_mode="liveall",
)

PARQUET_METADATA_DISK_USAGE = Gauge(
    name="parquet_metadata_disk_usage",
    documentation="Usage of the disk where the parquet metadata are stored (workers, used by /rows)",
    labelnames=["type"],
    multiprocess_mode="liveall",
)
METHOD_STEPS_PROCESSING_TIME = Histogram(
    "method_steps_processing_time_seconds",
    "Histogram of the processing time of specific steps in methods for a given context (in seconds)",
    ["method", "step"],
)
METHOD_LONG_STEPS_PROCESSING_TIME = Histogram(
    "method_long_steps_processing_time_seconds",
    "Histogram of the processing time of specific long steps in methods for a given context (in seconds)",
    ["method", "step"],
    buckets=LONG_DURATION_PROMETHEUS_HISTOGRAM_BUCKETS,
)


def update_queue_jobs_total() -> None:
    for job_metric in JobTotalMetricDocument.objects():
        QUEUE_JOBS_TOTAL.labels(
            queue=job_metric.job_type, status=job_metric.status, dataset_status=job_metric.dataset_status
        ).set(job_metric.total)


def update_worker_size_jobs_count() -> None:
    for jobs_count in WorkerSizeJobsCountDocument.objects():
        WORKER_SIZE_JOBS_COUNT.labels(worker_size=jobs_count.worker_size.value).set(jobs_count.jobs_count)


def update_responses_in_cache_total() -> None:
    for cache_metric in CacheTotalMetricDocument.objects():
        RESPONSES_IN_CACHE_TOTAL.labels(
            kind=cache_metric.kind, http_status=cache_metric.http_status, error_code=cache_metric.error_code
        ).set(cache_metric.total)


def update_disk_gauge(gauge: Gauge, directory: StrPath) -> None:
    # TODO: move to metrics, as for the other metrics (queue, cache)
    total, used, free, percent = disk_usage(str(directory))
    gauge.labels(type="total").set(total)
    gauge.labels(type="used").set(used)
    gauge.labels(type="free").set(free)
    gauge.labels(type="percent").set(percent)


def update_parquet_metadata_disk_usage(directory: StrPath) -> None:
    update_disk_gauge(PARQUET_METADATA_DISK_USAGE, directory)


T = TypeVar("T", bound="StepProfiler")


class StepProfiler:
    """
    A context manager that measures the time spent in a step of a method and reports it to Prometheus.

    Example:
        >>> with StepProfiler("method", "step") as profiler:
        ...     pass

    Args:
        method (`str`): The name of the method.
        step (`str`): The name of the step.
    """

    def __init__(self, method: str, step: str, histogram: Optional[Histogram] = None):
        self.histogram = METHOD_STEPS_PROCESSING_TIME if histogram is None else histogram
        self.method = method
        self.step = step
        self.before_time = time.perf_counter()

    def __enter__(self: T) -> T:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        after_time = time.perf_counter()
        self.histogram.labels(
            method=self.method,
            step=self.step,
        ).observe(after_time - self.before_time)


class LongStepProfiler(StepProfiler):
    def __init__(self, method: str, step: str):
        super().__init__(method, step, histogram=METHOD_LONG_STEPS_PROCESSING_TIME)
