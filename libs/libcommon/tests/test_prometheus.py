import os
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional

from libcommon.prometheus import (
    QUEUE_JOBS_TOTAL,
    RESPONSES_IN_CACHE_TOTAL,
    WORKER_SIZE_JOBS_COUNT,
    LongStepProfiler,
    Prometheus,
    StepProfiler,
    update_queue_jobs_total,
    update_responses_in_cache_total,
    update_worker_size_jobs_count,
)
from libcommon.queue.dataset_blockages import DATASET_STATUS_NORMAL
from libcommon.queue.metrics import JobTotalMetricDocument, WorkerSizeJobsCountDocument
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CacheTotalMetricDocument


def parse_metrics(content: str) -> dict[str, float]:
    # examples:
    # starlette_requests_total{method="GET",path_template="/metrics"} 1.0
    # method_steps_processing_time_seconds_sum{method="healthcheck_endpoint",step="all"} 1.6772013623267412e-05
    return {
        parts[0]: float(parts[1])
        for line in content.split("\n")
        if line and line[0] != "#" and (parts := line.rsplit(" ", 1))
    }


def test_prometheus() -> None:
    is_multiprocess = "PROMETHEUS_MULTIPROC_DIR" in os.environ

    prometheus = Prometheus()
    registry = prometheus.getRegistry()
    assert registry is not None

    content = prometheus.getLatestContent()
    metrics = parse_metrics(content)

    name = "process_start_time_seconds"
    if not is_multiprocess:
        assert name in metrics, metrics
        assert metrics[name] > 0, metrics[name]
    else:
        assert name not in metrics, metrics


def create_key(suffix: str, labels: dict[str, str], le: Optional[str] = None) -> str:
    items = list(labels.items())
    if le:
        items.append(("le", le))
    labels_string = ",".join([f'{key}="{value}"' for key, value in sorted(items)])
    return f"method_steps_processing_time_seconds_{suffix}{{{labels_string}}}"


def check_histogram_metric(
    metrics: dict[str, float],
    method: str,
    step: str,
    events: int,
    duration: float,
) -> None:
    labels = {"method": method, "step": step}
    assert metrics[create_key("count", labels)] == events, metrics
    assert metrics[create_key("bucket", labels, le="+Inf")] == events, metrics
    assert metrics[create_key("bucket", labels, le="1.0")] == events, metrics
    assert metrics[create_key("bucket", labels, le="0.05")] == 0, metrics
    assert metrics[create_key("sum", labels)] >= duration, metrics
    assert metrics[create_key("sum", labels)] <= duration * 1.1, metrics


def test_step_profiler() -> None:
    duration = 0.1
    method = "test_step_profiler"
    step_all = "all"
    with StepProfiler(method=method, step=step_all):
        time.sleep(duration)
    metrics = parse_metrics(Prometheus().getLatestContent())
    check_histogram_metric(metrics=metrics, method=method, step=step_all, events=1, duration=duration)


def test_step_profiler_with_custom_buckets() -> None:
    duration = 0.1
    method = "test_step_profiler"
    step_all = "all"
    with LongStepProfiler(method=method, step=step_all):
        time.sleep(duration)
    metrics = parse_metrics(Prometheus().getLatestContent())
    check_histogram_metric(metrics=metrics, method=method, step=step_all, events=1, duration=duration)


def test_nested_step_profiler() -> None:
    method = "test_nested_step_profiler"
    step_all = "all"
    step_1 = "step_1"
    duration_1a = 0.1
    duration_1b = 0.3
    step_2 = "step_2"
    duration_2 = 0.5
    with StepProfiler(method=method, step=step_all):
        with StepProfiler(method, step_1):
            time.sleep(duration_1a)
        with StepProfiler(method, step_1):
            time.sleep(duration_1b)
        with StepProfiler(method, step_2):
            time.sleep(duration_2)
    metrics = parse_metrics(Prometheus().getLatestContent())
    check_histogram_metric(
        metrics=metrics,
        method=method,
        step=step_all,
        events=1,
        duration=duration_1a + duration_1b + duration_2,
    )
    check_histogram_metric(metrics=metrics, method=method, step=step_1, events=2, duration=duration_1a + duration_1b)
    check_histogram_metric(metrics=metrics, method=method, step=step_2, events=1, duration=duration_2)


@dataclass
class Metrics:
    metrics: dict[str, float]

    def forge_metric_key(self, name: str, content: dict[str, str]) -> str:
        local_content: dict[str, str] = dict(content)
        if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
            local_content["pid"] = str(os.getpid())
        inner = ",".join([f'{key}="{value}"' for key, value in sorted(local_content.items())])
        return f"{name}{{{inner}}}"


def get_metrics() -> Metrics:
    prometheus = Prometheus()
    registry = prometheus.getRegistry()
    assert registry is not None
    content = prometheus.getLatestContent()
    lines = content.split("\n")
    metrics = {" ".join(line.split(" ")[:-1]): float(line.split(" ")[-1]) for line in lines if line and line[0] != "#"}
    return Metrics(metrics=metrics)


def test_cache_metrics(cache_mongo_resource: CacheMongoResource) -> None:
    RESPONSES_IN_CACHE_TOTAL.clear()

    cache_metric = {
        "kind": "dummy",
        "http_status": HTTPStatus.OK,
        "error_code": None,
        "total": 1,
    }

    collection = CacheTotalMetricDocument._get_collection()
    collection.insert_one(cache_metric)

    metrics = get_metrics()
    key = metrics.forge_metric_key(
        name="responses_in_cache_total",
        content={"error_code": "None", "http_status": "200", "kind": "dummy"},
    )
    assert key not in metrics.metrics

    update_responses_in_cache_total()

    metrics = get_metrics()
    assert key in metrics.metrics


def test_queue_metrics(queue_mongo_resource: QueueMongoResource) -> None:
    QUEUE_JOBS_TOTAL.clear()

    job_metric = {
        "job_type": "dummy",
        "status": "waiting",
        "total": 1,
    }

    collection = JobTotalMetricDocument._get_collection()
    collection.insert_one(job_metric)

    metrics = get_metrics()
    key = metrics.forge_metric_key(
        name="queue_jobs_total",
        content={"queue": "dummy", "status": "waiting", "dataset_status": DATASET_STATUS_NORMAL},
    )
    assert key not in metrics.metrics

    update_queue_jobs_total()

    metrics = get_metrics()
    assert key in metrics.metrics


def test_worker_size_jobs_count(queue_mongo_resource: QueueMongoResource) -> None:
    WORKER_SIZE_JOBS_COUNT.clear()

    metric = {
        "worker_size": "heavy",
        "jobs_count": 1,
    }

    collection = WorkerSizeJobsCountDocument._get_collection()
    collection.insert_one(metric)

    metrics = get_metrics()
    key = metrics.forge_metric_key(
        name="worker_size_jobs_count",
        content={"worker_size": "heavy"},
    )
    assert key not in metrics.metrics

    update_worker_size_jobs_count()

    metrics = get_metrics()
    assert key in metrics.metrics


def test_process_metrics() -> None:
    metrics = get_metrics()

    name = "process_start_time_seconds"

    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        assert name not in metrics.metrics
    else:
        assert name in metrics.metrics
        assert metrics.metrics[name] > 0
