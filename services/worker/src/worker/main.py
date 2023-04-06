# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
import os
import tempfile

from libcommon.log import init_logging
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.storage import init_assets_dir

from worker.config import AppConfig
from worker.executor import WorkerExecutor
from worker.job_runner_factory import JobRunnerFactory
from worker.resources import LibrariesResource

WORKER_STATE_FILE_NAME = "worker_state.json"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_file_path = os.path.join(tmp_dir, WORKER_STATE_FILE_NAME)
        os.environ["WORKER_STATE_FILE_PATH"] = state_file_path

        app_config = AppConfig.from_env()

        init_logging(level=app_config.log.level)
        # ^ set first to have logs as soon as possible
        assets_directory = init_assets_dir(directory=app_config.assets.storage_directory)

        processing_graph = ProcessingGraph(app_config.processing_graph.specification)

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
                processing_graph=processing_graph,
                hf_datasets_cache=libraries_resource.hf_datasets_cache,
                assets_directory=assets_directory,
            )
            worker_executor = WorkerExecutor(
                app_config=app_config,
                job_runner_factory=job_runner_factory,
                state_file_path=state_file_path,
            )
            worker_executor.start()
