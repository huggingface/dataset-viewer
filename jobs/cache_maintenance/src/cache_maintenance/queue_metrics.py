# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.metrics import JobTotalMetricDocument
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue


def collect_queue_metrics(processing_graph: ProcessingGraph) -> None:
    logging.info("collecting jobs metrics")
    queue = Queue()
    for processing_step in processing_graph.get_processing_steps():
        for status, total in queue.get_jobs_count_by_status(job_type=processing_step.job_type).items():
            JobTotalMetricDocument.objects(queue=processing_step.job_type, status=status).upsert_one(total=total)
