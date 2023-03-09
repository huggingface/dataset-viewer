import os
import time
from typing import Dict

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


def check_histogram_metric(metrics: Dict[str, float], method: str, step: str, events: int, duration: float) -> None:
    name = "method_steps_processing_time_seconds"
    params = f'method="{method}",step="{step}"'
    assert metrics[f"{name}_count{{{params}}}"] == events, metrics
    assert metrics[f'{name}_bucket{{le="+Inf",{params}}}'] == events, metrics
    assert metrics[f'{name}_bucket{{le="1.0",{params}}}'] == events, metrics
    assert metrics[f'{name}_bucket{{le="0.05",{params}}}'] == 0, metrics
    assert metrics[f"{name}_sum{{{params}}}"] >= duration, metrics
    assert metrics[f"{name}_sum{{{params}}}"] <= duration * 1.1, metrics


def test_step_profiler() -> None:
    duration = 0.1
    method = "test"
    step_all = "all"
    with StepProfiler(method=method, step=step_all):
        time.sleep(duration)
    metrics = parse_metrics(Prometheus().getLatestContent())
    check_histogram_metric(metrics=metrics, method=method, step=step_all, events=1, duration=duration)


def test_nested_step_profiler() -> None:
    method = "test"
    step_all = "all"
    step_1 = "step_1"
    duration_1a = 0.1
    duration_1b = 0.3
    step_2 = "step_2"
    duration_2 = 0.5
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
        metrics=metrics, method=method, step=step_all, events=1, duration=duration_1a + duration_1b + duration_2
    )
    check_histogram_metric(metrics=metrics, method=method, step=step_1, events=2, duration=duration_1a + duration_1b)
    check_histogram_metric(metrics=metrics, method=method, step=step_2, events=1, duration=duration_2)
