# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import sys

from libcommon.log import init_logging
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import QueueMongoResource

from cache_monitor.backfill import backfill_cache
from cache_monitor.config import JobConfig


def run_job() -> None:
    job_config = JobConfig.from_env()

    init_logging(level=job_config.log.level)

    with (
        QueueMongoResource(
            database=job_config.queue.mongo_database, host=job_config.queue.mongo_url
        ) as queue_resource,
    ):
        if not queue_resource.is_available():
            logging.warning(
                "The connection to the queue database could not be established. The cache refresh job is skipped."
            )
            return

        processing_graph = ProcessingGraph(job_config.graph.specification)
        init_processing_steps = processing_graph.get_first_steps()

        # TODO: support passing params to invoke different actions
        backfill_cache(
            init_processing_steps=init_processing_steps,
            hf_endpoint=job_config.common.hf_endpoint,
            hf_token=job_config.common.hf_token,
        )


if __name__ == "__main__":
    try:
        run_job()
        sys.exit(0)
    except Exception:
        sys.exit(1)
