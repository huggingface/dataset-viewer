# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os
from typing import List

from libcommon.processing_graph import ProcessingStep

from admin.prometheus import Prometheus


def test_prometheus(
    assets_directory: str,
    processing_steps: List[ProcessingStep],
) -> None:
    is_multiprocess = "PROMETHEUS_MULTIPROC_DIR" in os.environ

    prometheus = Prometheus(processing_steps=processing_steps, assets_storage_directory=assets_directory)
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

    for type in ["total", "used", "free", "percent"]:
        assert "assets_disk_usage{" + additional_field + 'type="' + type + '"}' in metrics
        assert metrics["assets_disk_usage{" + additional_field + 'type="' + type + '"}'] >= 0
    assert metrics["assets_disk_usage{" + additional_field + 'type="percent"}'] <= 100
