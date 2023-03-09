import os
import time
from typing import Dict, Optional

from api.prometheus import Prometheus, StepProfiler


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


def create_key(suffix: str, labels: Dict[str, str], le: Optional[str] = None) -> str:
    items = list(labels.items())
    if le:
        items.append(("le", le))
    labels_string = ",".join([f'{key}="{value}"' for key, value in sorted(items)])
    return f"method_steps_processing_time_seconds_{suffix}{{{labels_string}}}"


def check_histogram_metric(
    metrics: Dict[str, float], method: str, step: str, context: str, events: int, duration: float
) -> None:
    labels = {"context": context, "method": method, "step": step}
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
    context = "None"
    with StepProfiler(method=method, step=step_all):
        time.sleep(duration)
    metrics = parse_metrics(Prometheus().getLatestContent())
    check_histogram_metric(metrics=metrics, method=method, step=step_all, context=context, events=1, duration=duration)


def test_nested_step_profiler() -> None:
    method = "test_nested_step_profiler"
    step_all = "all"
    context = "None"
    step_1 = "step_1"
    duration_1a = 0.1
    duration_1b = 0.3
    context_1 = "None"
    step_2 = "step_2"
    duration_2 = 0.5
    context_2 = "endpoint: /splits"
    with StepProfiler(method=method, step=step_all):
        with StepProfiler("test", step_1):
            time.sleep(duration_1a)
        with StepProfiler("test", step_1):
            time.sleep(duration_1b)
        with StepProfiler("test", step_2):
            time.sleep(duration_2)
    metrics = parse_metrics(Prometheus().getLatestContent())
    print(metrics)
    check_histogram_metric(
        metrics=metrics,
        method=method,
        step=step_all,
        context=context,
        events=1,
        duration=duration_1a + duration_1b + duration_2,
    )
    check_histogram_metric(
        metrics=metrics, method=method, step=step_1, context=context_1, events=2, duration=duration_1a + duration_1b
    )
    check_histogram_metric(
        metrics=metrics, method=method, step=step_2, context=context_2, events=1, duration=duration_2
    )
