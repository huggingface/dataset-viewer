# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import sys
from datetime import datetime

from libcommon.log import init_logging
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import (
    CacheMongoResource,
    MetricsMongoResource,
    QueueMongoResource,
)

from cache_maintenance.backfill import backfill_cache
from cache_maintenance.config import JobConfig
from cache_maintenance.metrics import collect_metrics


def run_job() -> None:
    job_config = JobConfig.from_env()
    action = job_config.action
    supported_actions = ["backfill", "collect-metrics", "skip"]
    #  In the future we will support other kind of actions
    if not action:
        logging.warning("No action mode was selected, skipping tasks.")
        return
    if action not in supported_actions:
        logging.warning(f"Wrong action mode selected, supported actions are {supported_actions}.")
        return

    init_logging(level=job_config.log.level)

    with (
        CacheMongoResource(
            database=job_config.cache.mongo_database, host=job_config.cache.mongo_url
        ) as cache_resource,
        QueueMongoResource(
            database=job_config.queue.mongo_database, host=job_config.queue.mongo_url
        ) as queue_resource,
        MetricsMongoResource(
            database=job_config.metrics.mongo_database, host=job_config.metrics.mongo_url
        ) as metrics_resource,
    ):
        if not cache_resource.is_available():
            logging.warning("The connection to the cache database could not be established. The action is skipped.")
            return
        if not queue_resource.is_available():
            logging.warning("The connection to the queue database could not be established. The action is skipped.")
            return
        if not metrics_resource.is_available():
            logging.warning("The connection to the metrics database could not be established. The action is skipped.")
            return

        processing_graph = ProcessingGraph(job_config.graph.specification)
        start_time = datetime.now()

        if action == "backfill":
            backfill_cache(
                processing_graph=processing_graph,
                hf_endpoint=job_config.common.hf_endpoint,
                hf_token=job_config.common.hf_token,
                error_codes_to_retry=job_config.backfill.error_codes_to_retry,
            )
        elif action == "collect-metrics":
            collect_metrics(processing_graph=processing_graph)

        end_time = datetime.now()
        logging.info(f"Duration: {end_time - start_time}")


if __name__ == "__main__":
    try:
        run_job()
        sys.exit(0)
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
