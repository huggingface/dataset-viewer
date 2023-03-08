# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import sys

from libcommon.config import ProcessingGraphConfig
from libcommon.log import init_logging
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource

from cache_refresh.config import JobConfig
from cache_refresh.outdated_cache import refresh_cache


def run_job() -> None:
    job_config = JobConfig.from_env()

    init_logging(log_level=job_config.common.log_level)
    # ^ set first to have logs as soon as possible

    with (
        CacheMongoResource(
            database=job_config.cache.mongo_database, host=job_config.cache.mongo_url
        ) as cache_resource,
        QueueMongoResource(
            database=job_config.queue.mongo_database, host=job_config.queue.mongo_url
        ) as queue_resource,
    ):
        if not cache_resource.is_available():
            logging.warning(
                "The connection to the cache database could not be established. The cache refresh job is skipped."
            )
            return
        if not queue_resource.is_available():
            logging.warning(
                "The connection to the queue database could not be established. The cache refresh job is skipped."
            )
            return

        processing_graph_config = ProcessingGraphConfig.from_env()
        processing_graph = ProcessingGraph(processing_graph_config.specification)
        processing_steps = list(processing_graph.steps.values())
        refresh_cache(processing_steps, processing_graph_config)


if __name__ == "__main__":
    try:
        run_job()
        sys.exit(0)
    except Exception:
        sys.exit(1)
