# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Queue
from libcommon.simple_cache import get_responses_count_by_kind_status_and_error_code
from libcommon.metrics import CustomMetric


def collect_metrics(
    processing_steps: list[ProcessingStep]
) -> None:
    logging.info("collecting jobs metrics")
    queue = Queue()
    for processing_step in processing_steps:
        for status, total in queue.get_jobs_count_by_status(job_type=processing_step.job_type).items():
            CustomMetric(
                metric="queue_jobs_total",
                content={
                    "queue": processing_step.job_type,
                    "status": status,
                    "count": total
                }
            ).save()

    logging.info("collecting cache metrics")
    for metric in get_responses_count_by_kind_status_and_error_code():
        CustomMetric(
            metric="responses_in_cache_total",
            content={
                "kind": metric["kind"],
                "http_status": metric["http_status"],
                "error_code": metric["error_code"],
                "count": metric["count"]
            }
        ).save()
    logging.info("metrics have been collected")
