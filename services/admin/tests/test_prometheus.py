import os
from typing import List

from libcommon.processing_graph import ProcessingStep

from admin.config import AppConfig
from admin.prometheus import Prometheus


def test_prometheus(app_config: AppConfig, processing_steps: List[ProcessingStep]) -> None:
    # we depend on app_config to be sure we already connected to the database
    is_multiprocess = "PROMETHEUS_MULTIPROC_DIR" in os.environ

    prometheus = Prometheus(processing_steps=processing_steps)
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
    for processing_step in processing_steps:
        assert (
            "queue_jobs_total{" + additional_field + 'queue="' + processing_step.job_type + '",status="started"}'
            in metrics
        )
