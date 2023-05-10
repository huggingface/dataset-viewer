# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os
from http import HTTPStatus

from libcommon.metrics import CacheTotalMetric, JobTotalMetric
from libcommon.processing_graph import ProcessingGraph
from libcommon.storage import StrPath

from admin.prometheus import Prometheus


def test_prometheus(
    assets_directory: StrPath,
    processing_graph: ProcessingGraph,
) -> None:
    cache_metric = {
        "kind": "dummy",
        "http_status": HTTPStatus.OK,
        "error_code": None,
        "total": 1,
    }

    collection = CacheTotalMetric._get_collection()
    collection.insert_one(cache_metric)

    job_metric = {
        "queue": "dummy",
        "status": "waiting",
        "total": 1,
    }

    collection = JobTotalMetric._get_collection()
    collection.insert_one(job_metric)

    is_multiprocess = "PROMETHEUS_MULTIPROC_DIR" in os.environ

    prometheus = Prometheus(processing_graph=processing_graph, assets_directory=assets_directory)
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

    additional_field = f'pid="{os.getpid()}"' if is_multiprocess else ""
    last_additional_field = f",{additional_field}" if additional_field else ""
    not_last_additional_field = f"{additional_field}," if additional_field else ""

    assert (
        'responses_in_cache_total{error_code="None",http_status="200",kind="dummy"' + last_additional_field + "}"
        in metrics
    )
    assert "queue_jobs_total{" + not_last_additional_field + 'queue="dummy",status="waiting"}' in metrics

    for type in ["total", "used", "free", "percent"]:
        assert "assets_disk_usage{" + not_last_additional_field + 'type="' + type + '"}' in metrics
        assert metrics["assets_disk_usage{" + not_last_additional_field + 'type="' + type + '"}'] >= 0
    assert metrics["assets_disk_usage{" + not_last_additional_field + 'type="percent"}'] <= 100
