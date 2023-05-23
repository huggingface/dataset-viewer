# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os
import re
from typing import Mapping

from .utils import ADMIN_URL, get


def has_metric(name: str, labels: Mapping[str, str], metric_names: set[str]) -> bool:
    label_str = ",".join([f'{k}="{v}"' for k, v in sorted(labels.items())])
    s = name + "{" + label_str + "}"
    return any(re.match(s, metric_name) is not None for metric_name in metric_names)


def test_metrics() -> None:
    assert "PROMETHEUS_MULTIPROC_DIR" in os.environ
    response = get("/metrics", url=ADMIN_URL)
    assert response.status_code == 200, f"{response.status_code} - {response.text}"
    content = response.text
    lines = content.split("\n")
    metrics = {line.split(" ")[0]: float(line.split(" ")[1]) for line in lines if line and line[0] != "#"}
    # see https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn
    assert "process_start_time_seconds" not in metrics

    # the middleware should have recorded the request
    name = 'starlette_requests_total{method="GET",path_template="/metrics"}'
    assert name in metrics, metrics
    assert metrics[name] > 0, metrics

    metric_names = set(metrics.keys())

    # the cache and queue metrics are computed by the background jobs. Here, in the e2e tests, we don't run them,
    # so we should not see any of these metrics.

    for queue in ["/config-names", "split-first-rows-from-streaming", "dataset-parquet"]:
        # eg. 'queue_jobs_total{pid="10",queue="split-first-rows-from-streaming",status="started"}'
        assert not has_metric(
            name="queue_jobs_total",
            labels={"pid": "[0-9]*", "queue": queue, "status": "started"},
            metric_names=metric_names,
        ), f"queue_jobs_total - queue={queue} found in {metrics}"
    for cache_kind in ["/config-names", "split-first-rows-from-streaming", "dataset-parquet"]:
        # cache should have been filled by the previous tests
        # eg. 'responses_in_cache_total{error_code="None",http_status="200",path="/config-names",pid="10"}'
        assert not has_metric(
            name="responses_in_cache_total",
            labels={"error_code": "None", "http_status": "200", "kind": cache_kind, "pid": "[0-9]*"},
            metric_names=metric_names,
        ), f"responses_in_cache_total - cache kind {cache_kind} found in {metrics}"

    # the assets metrics, on the other end, are computed at runtime, so we should see them
    assert has_metric(
        name="assets_disk_usage",
        labels={"type": "total", "pid": "[0-9]*"},
        metric_names=metric_names,
    ), f"assets_disk_usage - cache kind {cache_kind} found in {metrics}"
