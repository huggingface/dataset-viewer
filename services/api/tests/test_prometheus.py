import os

from api.prometheus import Prometheus


def test_prometheus() -> None:
    is_multiprocess = "PROMETHEUS_MULTIPROC_DIR" in os.environ

    prometheus = Prometheus()
    registry = prometheus.getRegistry()
    assert registry is not None

    content = prometheus.getLatestContent()
    print("content:", content)
    lines = content.split("\n")
    # examples:
    # starlette_requests_total{method="GET",path_template="/metrics"} 1.0
    # method_steps_processing_time_seconds_sum{method="healthcheck_endpoint",step="all"} 1.6772013623267412e-05
    metrics = {
        parts[0]: float(parts[1]) for line in lines if line and line[0] != "#" and (parts := line.rsplit(" ", 1))
    }

    name = "process_start_time_seconds"
    if not is_multiprocess:
        assert name in metrics, metrics
        assert metrics[name] > 0, metrics[name]
    else:
        assert name not in metrics, metrics
