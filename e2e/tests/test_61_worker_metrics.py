# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import os

from .utils import WORKER_URL, get


def test_worker_metrics() -> None:
    assert "PROMETHEUS_MULTIPROC_DIR" in os.environ
    response = get("/metrics", url=WORKER_URL)
    assert response.status_code == 200, f"{response.status_code} - {response.text}"
    content = response.text
    lines = content.split("\n")
    assert "# TYPE method_steps_processing_time_seconds histogram" in lines
    # examples:
    # method_steps_processing_time_seconds_sum{method="healthcheck_endpoint",step="all"} 1.6772013623267412e-05
    metrics = {
        parts[0]: float(parts[1]) for line in lines if line and line[0] != "#" and (parts := line.rsplit(" ", 1))
    }
    assert "process_start_time_seconds" not in metrics
    # ^ see https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn
    assert "method_steps_processing_time_seconds_sum" in {metric.split("{")[0] for metric in metrics}
