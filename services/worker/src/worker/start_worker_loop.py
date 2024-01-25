# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import sys

from libcommon.log import init_logging
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.storage import (
    init_duckdb_index_cache_dir,
    init_parquet_metadata_dir,
    init_statistics_cache_dir,
)
from libcommon.storage_client import StorageClient

from worker.config import AppConfig
from worker.job_runner_factory import JobRunnerFactory
from worker.loop import Loop
from worker.resources import LibrariesResource

if __name__ == "__main__":
    app_config = AppConfig.from_env()

    state_file_path = app_config.worker.state_file_path
    if "--print-worker-state-path" in sys.argv:
        print(state_file_path, flush=True)
    if not state_file_path:
        raise RuntimeError("The worker state file path is not set. Exiting.")

    init_logging(level=app_config.log.level)
    # ^ set first to have logs as soon as possible
    parquet_metadata_directory = init_parquet_metadata_dir(directory=app_config.parquet_metadata.storage_directory)
    duckdb_index_cache_directory = init_duckdb_index_cache_dir(directory=app_config.duckdb_index.cache_directory)
    statistics_cache_directory = init_statistics_cache_dir(app_config.descriptive_statistics.cache_directory)

    storage_client = StorageClient(
        protocol=app_config.assets.storage_protocol,
        storage_root=app_config.assets.storage_root,
        base_url=app_config.assets.base_url,
        overwrite=True,  # all the job runners will overwrite the files
        s3_config=app_config.s3,
        # no need to specify cloudfront config here, as we are not generating signed urls in cached entries
    )

    with (
        LibrariesResource(
            hf_endpoint=app_config.common.hf_endpoint,
            init_hf_datasets_cache=app_config.datasets_based.hf_datasets_cache,
            numba_path=app_config.numba.path,
        ) as libraries_resource,
        CacheMongoResource(
            database=app_config.cache.mongo_database, host=app_config.cache.mongo_url
        ) as cache_resource,
        QueueMongoResource(
            database=app_config.queue.mongo_database, host=app_config.queue.mongo_url
        ) as queue_resource,
    ):
        if not cache_resource.is_available():
            raise RuntimeError("The connection to the cache database could not be established. Exiting.")
        if not queue_resource.is_available():
            raise RuntimeError("The connection to the queue database could not be established. Exiting.")

        job_runner_factory = JobRunnerFactory(
            app_config=app_config,
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
            parquet_metadata_directory=parquet_metadata_directory,
            duckdb_index_cache_directory=duckdb_index_cache_directory,
            statistics_cache_directory=statistics_cache_directory,
            storage_client=storage_client,
        )
        loop = Loop(
            job_runner_factory=job_runner_factory,
            state_file_path=state_file_path,
            app_config=app_config,
        )
        loop.run()
