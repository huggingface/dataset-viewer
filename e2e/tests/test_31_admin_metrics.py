# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os
import re
from typing import Dict

from .utils import ADMIN_URL, get


def has_metric(name: str, labels: Dict[str, str], metrics: set[str]) -> bool:
    labels = dict({"pid": "[0-9]*"}, **labels)
    label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
    s = name + "{" + label_str + "}"
    return any(re.match(s, metric) is not None for metric in metrics)


def test_metrics():
    assert "PROMETHEUS_MULTIPROC_DIR" in os.environ
    response = get("/metrics", url=ADMIN_URL)
    assert response.status_code == 200, f"{response.status_code} - {response.text}"
    content = response.text
    lines = content.split("\n")
    metrics = {line.split(" ")[0] for line in lines if line and line[0] != "#"}
    # see https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn
    assert "process_start_time_seconds" not in metrics

    for endpoint in ["/splits", "/first-rows"]:
        assert has_metric(name="queue_jobs_total", labels={"queue": endpoint, "status": "started"}, metrics=metrics)
        # cache should have been filled by the previous tests
        assert has_metric(
            name="responses_in_cache_total",
            labels={"path": endpoint, "http_status": "200", "error_code": "null"},
            metrics=metrics,
        )
