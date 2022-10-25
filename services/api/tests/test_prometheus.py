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
    metrics = {line.split(" ")[0]: float(line.split(" ")[1]) for line in lines if line and line[0] != "#"}
    name = "process_start_time_seconds"
    if not is_multiprocess:
        assert name in metrics, metrics
        assert metrics[name] > 0, metrics[name]
    else:
        assert name not in metrics, metrics
