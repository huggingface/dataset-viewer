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
            job_type = processing_step.job_type
            query_set = JobTotalMetricDocument.objects(job_type=job_type, status=status)
            current_metric = query_set.first()
            if current_metric is not None:
                current_total = current_metric.total
                logging.info(
                    f"{job_type=} {status=} current_total={current_total} new_total="
                    f"{new_total} difference={int(new_total)-current_total}"  # type: ignore
                )
            query_set.upsert_one(total=new_total)
    logging.info("queue metrics have been collected")
