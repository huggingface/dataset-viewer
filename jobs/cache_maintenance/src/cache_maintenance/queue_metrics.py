# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import JobTotalMetricDocument, Queue


def collect_queue_metrics(processing_graph: ProcessingGraph) -> None:
    logging.info("collecting queue metrics")
    queue = Queue()
    for processing_step in processing_graph.get_processing_steps():
        for status, new_total in queue.get_jobs_count_by_status(job_type=processing_step.job_type).items():
            JobTotalMetricDocument.objects(job_type=processing_step.job_type, status=status).upsert_one(
                total=new_total
            )
    logging.info("queue metrics have been collected")
