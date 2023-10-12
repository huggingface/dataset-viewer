# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import sys
from datetime import datetime

from libcommon.log import init_logging
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.storage import init_dir

from cache_maintenance.backfill import backfill_cache
from cache_maintenance.cache_metrics import collect_cache_metrics
from cache_maintenance.clean_directory import clean_directory
from cache_maintenance.config import JobConfig
from cache_maintenance.discussions import post_messages
from cache_maintenance.queue_metrics import collect_queue_metrics


def run_job() -> None:
    job_config = JobConfig.from_env()
    action = job_config.action
    #  In the future we will support other kind of actions
    if not action:
        logging.warning("No action mode was selected, skipping tasks.")
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
        processing_graph = ProcessingGraph(job_config.graph)
        start_time = datetime.now()
        if action == "backfill":
            if not cache_resource.is_available():
                logging.warning(
                    "The connection to the cache database could not be established. The action is skipped."
                )
                return
            if not queue_resource.is_available():
                logging.warning(
                    "The connection to the queue database could not be established. The action is skipped."
                )
                return
            backfill_cache(
                processing_graph=processing_graph,
                hf_endpoint=job_config.common.hf_endpoint,
                hf_token=job_config.common.hf_token,
                blocked_datasets=job_config.common.blocked_datasets,
                error_codes_to_retry=job_config.backfill.error_codes_to_retry,
                cache_max_days=job_config.cache.max_days,
            )
        elif action == "clean-directory":
            directory_path = init_dir(directory=job_config.directory_cleaning.cache_directory)
            folder_pattern = f"{directory_path}/{job_config.directory_cleaning.subfolder_pattern}"
            clean_directory(
                pattern=folder_pattern,
                expired_time_interval_seconds=job_config.directory_cleaning.expired_time_interval_seconds,
            )
        elif action == "collect-queue-metrics":
            if not queue_resource.is_available():
                logging.warning(
                    "The connection to the queue database could not be established. The action is skipped."
                )
                return
            collect_queue_metrics(processing_graph=processing_graph)
        elif action == "collect-cache-metrics":
            if not cache_resource.is_available():
                logging.warning(
                    "The connection to the cache database could not be established. The action is skipped."
                )
                return
            collect_cache_metrics()
        elif action == "post-messages":
            if not cache_resource.is_available():
                logging.warning(
                    "The connection to the cache database could not be established. The action is skipped."
                )
                return
            post_messages(
                hf_endpoint=job_config.common.hf_endpoint,
                bot_associated_user_name=job_config.discussions.bot_associated_user_name,
                bot_token=job_config.discussions.bot_token,
                parquet_revision=job_config.discussions.parquet_revision,
            )
        elif action == "skip":
            pass
        else:
            logging.warning(f"Action '{action}' is not supported.")

        end_time = datetime.now()
        logging.info(f"Duration: {end_time - start_time}")


if __name__ == "__main__":
    try:
        run_job()
        sys.exit(0)
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
