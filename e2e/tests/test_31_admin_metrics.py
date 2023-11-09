# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os

from .utils import ADMIN_URL, get, has_metric


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

    # the queue metrics are computed each time a job is created and processed
    # they should exists at least for some of jobs types
    for queue in ["dataset-config-names", "split-first-rows-from-streaming", "dataset-parquet"]:
        # eg. 'queue_jobs_total{pid="10",queue="split-first-rows-from-streaming",status="started"}'
        assert has_metric(
            name="queue_jobs_total",
            labels={"pid": "[0-9]*", "queue": queue, "status": "started"},
            metric_names=metric_names,
        ), f"queue_jobs_total - queue={queue} found in {metrics}"

    # the cache metrics are computed each time a job is processed
    # they should exists at least for some of cache kinds
    for cache_kind in ["dataset-config-names", "split-first-rows-from-streaming", "dataset-parquet"]:
        # cache should have been filled by the previous tests
        # eg. 'responses_in_cache_total{error_code="None",http_status="200",path="dataset-config-names",pid="10"}'
        assert has_metric(
            name="responses_in_cache_total",
            labels={"error_code": "None", "http_status": "200", "kind": cache_kind, "pid": "[0-9]*"},
            metric_names=metric_names,
        ), f"responses_in_cache_total - cache kind {cache_kind} found in {metrics}"

    # the disk usage metrics, on the other end, are computed at runtime, so we should see them
    assert has_metric(
        name="descriptive_statistics_disk_usage",
        labels={"type": "total", "pid": "[0-9]*"},
        metric_names=metric_names,
    ), "descriptive_statistics_disk_usage"

    assert has_metric(
        name="duckdb_disk_usage",
        labels={"type": "total", "pid": "[0-9]*"},
        metric_names=metric_names,
    ), "duckdb_disk_usage"

    assert has_metric(
        name="hf_datasets_disk_usage",
        labels={"type": "total", "pid": "[0-9]*"},
        metric_names=metric_names,
    ), "hf_datasets_disk_usage"

    assert has_metric(
        name="parquet_metadata_disk_usage",
        labels={"type": "total", "pid": "[0-9]*"},
        metric_names=metric_names,
    ), "parquet_metadata_disk_usage"
