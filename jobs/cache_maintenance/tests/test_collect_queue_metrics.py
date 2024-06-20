# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from unittest.mock import patch

import pytest
from libcommon.new_queue.metrics import JobTotalMetricDocument, WorkerSizeJobsCountDocument
from libcommon.queue import JobsCountByWorkerSize, JobsTotalByTypeAndStatus, Queue

from cache_maintenance.queue_metrics import collect_queue_metrics, collect_worker_size_jobs_count

JOB_TYPE_A = "JobTypeA"
STATUS_WAITING = "waiting"
COUNT = 1

NEW_METRIC = {(JOB_TYPE_A, STATUS_WAITING): COUNT}
OTHER_JOB_TYPE = {("JobTypeB", STATUS_WAITING): COUNT}
OTHER_STATUS = {(JOB_TYPE_A, "started"): COUNT}
OTHER_COUNT = {(JOB_TYPE_A, STATUS_WAITING): COUNT + 1}


WORKER_SIZE = "medium"
NEW_WORKER_SIZE_METRIC = {WORKER_SIZE: COUNT}
OTHER_WORKER_SIZE = {"heavy": COUNT}
OTHER_WORKER_SIZE_COUNT = {WORKER_SIZE: COUNT + 1}


class MockQueue(Queue):
    def get_jobs_total_by_type_and_status(self) -> JobsTotalByTypeAndStatus:
        return NEW_METRIC

    def get_jobs_count_by_worker_size(self) -> JobsCountByWorkerSize:
        return NEW_WORKER_SIZE_METRIC


@pytest.mark.parametrize(
    "old_metrics",
    [{}, NEW_METRIC, OTHER_JOB_TYPE, OTHER_STATUS, OTHER_COUNT],
)
def test_collect_jobs_metrics(old_metrics: JobsTotalByTypeAndStatus) -> None:
    for (job_type, status), total in old_metrics.items():
        JobTotalMetricDocument(job_type=job_type, status=status, total=total).save()

    with patch(
        "cache_maintenance.queue_metrics.Queue",
        wraps=MockQueue,
    ):
        collect_queue_metrics()

    assert {
        (metric.job_type, metric.status): metric.total for metric in JobTotalMetricDocument.objects()
    } == NEW_METRIC


@pytest.mark.parametrize(
    "old_metrics",
    [{}, NEW_WORKER_SIZE_METRIC, OTHER_WORKER_SIZE, OTHER_WORKER_SIZE_COUNT],
)
def test_collect_worker_size_jobs_count(old_metrics: JobsCountByWorkerSize) -> None:
    for worker_size, jobs_count in old_metrics.items():
        WorkerSizeJobsCountDocument(worker_size=worker_size, jobs_count=jobs_count).save()

    with patch(
        "cache_maintenance.queue_metrics.Queue",
        wraps=MockQueue,
    ):
        collect_worker_size_jobs_count()

    assert {
        metric.worker_size: metric.jobs_count for metric in WorkerSizeJobsCountDocument.objects()
    } == NEW_WORKER_SIZE_METRIC
