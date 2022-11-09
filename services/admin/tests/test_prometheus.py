import os

from admin.config import AppConfig
from admin.prometheus import Prometheus
from admin.utils import JobType


def test_prometheus(app_config: AppConfig) -> None:
    # we depend on app_config to be sure we already connected to the database
    is_multiprocess = "PROMETHEUS_MULTIPROC_DIR" in os.environ

    prometheus = Prometheus()
    registry = prometheus.getRegistry()
    assert registry is not None

    content = prometheus.getLatestContent()
    print("content:", content)
    lines = content.split("\n")
    metrics = {line.split(" ")[0]: float(line.split(" ")[1]) for line in lines if line and line[0] != "#"}

    name = "process_start_time_seconds"
    if is_multiprocess:
        assert name not in metrics
    else:
        assert name in metrics
        assert metrics[name] > 0

    additional_field = ('pid="' + str(os.getpid()) + '",') if is_multiprocess else ""
    for _, job_type in JobType.__members__.items():
        assert "queue_jobs_total{" + additional_field + 'queue="' + job_type.value + '",status="started"}' in metrics
