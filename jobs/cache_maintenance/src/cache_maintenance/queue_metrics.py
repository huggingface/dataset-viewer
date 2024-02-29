# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.queue import JobTotalMetricDocument, Queue, WorkerSizeJobsCountDocument


def collect_queue_metrics() -> None:
    """
    Collects queue metrics and updates the queue metrics in the database.

    The obsolete queue metrics are deleted, and the new ones are inserted or updated.

    We don't delete everything, then create everything, because the /metrics endpoint could be called at the same time,
    and the metrics would be inconsistent.
    """
    logging.info("updating queue metrics")

    new_metric_by_id = Queue().get_jobs_total_by_type_and_status()
    new_ids = set(new_metric_by_id.keys())
    old_ids = set((metric.job_type, metric.status) for metric in JobTotalMetricDocument.objects())
    to_delete = old_ids - new_ids

    for job_type, status in to_delete:
        JobTotalMetricDocument.objects(job_type=job_type, status=status).delete()
        logging.info(f"{job_type=} {status=} has been deleted")

    for (job_type, status), total in new_metric_by_id.items():
        JobTotalMetricDocument.objects(job_type=job_type, status=status).upsert_one(total=total)
        logging.info(f"{job_type=} {status=}: {total=} has been inserted")

    logging.info("queue metrics have been updated")


def collect_worker_size_jobs_count() -> None:
    """
    Collects worker_size_jobs_count metrics and updates them in the database.

    The obsolete metrics are deleted, and the new ones are inserted or updated.

    We don't delete everything, then create everything, because the /metrics endpoint could be called at the same time,
    and the metrics would be inconsistent.
    """
    logging.info("updating worker_size_jobs_count metrics")

    new_metric_by_worker_size = Queue().get_jobs_count_by_worker_size()
    new_ids = set(worker_size for worker_size in new_metric_by_worker_size.keys())
    old_ids = set(metric.worker_size.value for metric in WorkerSizeJobsCountDocument.objects())
    to_delete = old_ids - new_ids

    for worker_size in to_delete:
        WorkerSizeJobsCountDocument.objects(worker_size=worker_size).delete()
        logging.info(f"{worker_size=} has been deleted")

    for worker_size, jobs_count in new_metric_by_worker_size.items():
        WorkerSizeJobsCountDocument.objects(worker_size=worker_size).upsert_one(jobs_count=jobs_count)
        logging.info(f"{worker_size=}: {jobs_count=} has been inserted")

    logging.info("worker_size_jobs_count metrics have been updated")
