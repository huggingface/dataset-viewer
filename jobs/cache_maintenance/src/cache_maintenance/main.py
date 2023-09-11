# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import sys
from datetime import datetime

from libcommon.log import init_logging
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.storage import init_duckdb_index_cache_dir

from cache_maintenance.backfill import backfill_cache
from cache_maintenance.cache_metrics import collect_cache_metrics
from cache_maintenance.config import JobConfig
from cache_maintenance.delete_indexes import delete_indexes
from cache_maintenance.queue_metrics import collect_queue_metrics


def run_job() -> None:
    job_config = JobConfig.from_env()
    action = job_config.action
    supported_actions = ["backfill", "collect-cache-metrics", "collect-queue-metrics", "delete-indexes", "skip"]
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
    ):
        if not cache_resource.is_available():
            logging.warning("The connection to the cache database could not be established. The action is skipped.")
            return
        if not queue_resource.is_available():
            logging.warning("The connection to the queue database could not be established. The action is skipped.")
            return

        processing_graph = ProcessingGraph(job_config.graph.specification)
        start_time = datetime.now()
        if action == "backfill":
            backfill_cache(
                processing_graph=processing_graph,
                hf_endpoint=job_config.common.hf_endpoint,
                hf_token=job_config.common.hf_token,
                error_codes_to_retry=job_config.backfill.error_codes_to_retry,
                cache_max_days=job_config.cache.max_days,
            )
        elif action == "collect-queue-metrics":
            collect_queue_metrics(processing_graph=processing_graph)
        elif action == "collect-cache-metrics":
            collect_cache_metrics()
        elif action == "delete-indexes":
            duckdb_index_cache_directory = init_duckdb_index_cache_dir(directory=job_config.duckdb.cache_directory)
            delete_indexes(
                duckdb_index_cache_directory=duckdb_index_cache_directory,
                subdirectory=job_config.duckdb.subdirectory,
                expired_time_interval_seconds=job_config.duckdb.expired_time_interval_seconds,
                file_extension=job_config.duckdb.file_extension,
            )

        end_time = datetime.now()
        logging.info(f"Duration: {end_time - start_time}")


if __name__ == "__main__":
    try:
        run_job()
        sys.exit(0)
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
