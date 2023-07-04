# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os
import re
from typing import Mapping

from .utils import ROWS_URL, get


def has_metric(name: str, labels: Mapping[str, str], metric_names: set[str]) -> bool:
    label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
    s = name + "{" + label_str + "}"
    return any(re.match(s, metric_name) is not None for metric_name in metric_names)


def test_metrics() -> None:
    assert "PROMETHEUS_MULTIPROC_DIR" in os.environ
    response = get("/metrics", url=ROWS_URL)
    assert response.status_code == 200, f"{response.status_code} - {response.text}"
    content = response.text
    lines = content.split("\n")
    # examples:
    # starlette_requests_total{method="GET",path_template="/metrics"} 1.0
    # method_steps_processing_time_seconds_sum{method="healthcheck_endpoint",step="all"} 1.6772013623267412e-05
    metrics = {
        parts[0]: float(parts[1]) for line in lines if line and line[0] != "#" and (parts := line.rsplit(" ", 1))
    }
    # see https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn
    assert "process_start_time_seconds" not in metrics

    # the middleware should have recorded the request
    name = 'starlette_requests_total{method="GET",path_template="/metrics"}'
    assert name in metrics, metrics
    assert metrics[name] > 0, metrics

    metric_names = set(metrics.keys())
    for endpoint in ["/rows"]:
        # these metrics are only available in the admin API
        assert not has_metric(
            name="queue_jobs_total",
            labels={"pid": "[0-9]*", "queue": endpoint, "status": "started"},
            metric_names=metric_names,
        ), f"queue_jobs_total - endpoint={endpoint} found in {metrics}"
        assert not has_metric(
            name="responses_in_cache_total",
            labels={"error_code": "None", "http_status": "200", "path": endpoint, "pid": "[0-9]*"},
            metric_names=metric_names,
        ), f"responses_in_cache_total - endpoint {endpoint} found in {metrics}"
