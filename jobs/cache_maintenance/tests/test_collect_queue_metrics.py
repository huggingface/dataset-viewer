# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.dtos import Status
from libcommon.processing_graph import processing_graph
from libcommon.queue import JobTotalMetricDocument, Queue

from cache_maintenance.queue_metrics import collect_queue_metrics


def test_collect_queue_metrics() -> None:
    queue = Queue()
    queue.add_job(
        job_type="dataset-config-names",
        dataset="dataset",
        revision="revision",
        config="config",
        split="split",
        difficulty=50,
    )
    assert JobTotalMetricDocument.objects().count() == 1

    collect_queue_metrics()

    job_metrics = JobTotalMetricDocument.objects().all()
    assert job_metrics
    assert len(job_metrics) == len(Status) * len(
        processing_graph.get_processing_steps()
    )  # One per job status and per step, see libcommon.queue.get_jobs_count_by_status
    waiting_job = next((job for job in job_metrics if job.status == "waiting"), None)
    assert waiting_job
    assert waiting_job.total == 1

    remaining_status = [job for job in job_metrics if job.status != "waiting"]
    assert remaining_status
    assert all(job.total == 0 for job in remaining_status)
