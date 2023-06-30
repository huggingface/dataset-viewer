# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.metrics import CacheTotalMetricDocument, JobTotalMetricDocument
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue
from libcommon.simple_cache import get_responses_count_by_kind_status_and_error_code


def collect_metrics(processing_graph: ProcessingGraph) -> None:
    logging.info("collecting jobs metrics")
    queue = Queue()
    for processing_step in processing_graph.get_processing_steps():
        for status, total in queue.get_jobs_count_by_status(job_type=processing_step.job_type).items():
            JobTotalMetricDocument.objects(queue=processing_step.job_type, status=status).upsert_one(total=total)

    logging.info("collecting cache metrics")
    for metric in get_responses_count_by_kind_status_and_error_code():
        CacheTotalMetricDocument.objects(
            kind=metric["kind"], http_status=metric["http_status"], error_code=metric["error_code"]
        ).upsert_one(total=metric["count"])
    logging.info("metrics have been collected")
