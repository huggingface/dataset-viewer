# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import JobTotalMetricDocument, Queue
from libcommon.utils import Status

from cache_maintenance.queue_metrics import collect_queue_metrics


def test_collect_queue_metrics() -> None:
    processing_step_name = "test_type"
    processing_graph = ProcessingGraph(
        processing_graph_specification={processing_step_name: {"input_type": "dataset", "job_runner_version": 1}}
    )
    processing_step = processing_graph.get_processing_step(processing_step_name)
    queue = Queue()
    queue.add_job(
        job_type=processing_step.job_type,
        dataset="dataset",
        revision="revision",
        config="config",
        split="split",
        difficulty=50,
    )

    collect_queue_metrics(processing_graph=processing_graph)

    job_metrics = JobTotalMetricDocument.objects()
    assert job_metrics
    assert len(job_metrics) == len(Status)  # One by each job state, see libcommon.queue.get_jobs_count_by_status
    waiting_job = next((job for job in job_metrics if job.status == "waiting"), None)
    assert waiting_job
    assert waiting_job.total == 1

    remaining_status = [job for job in job_metrics if job.status != "waiting"]
    assert remaining_status
    assert all(job.total == 0 for job in remaining_status)
