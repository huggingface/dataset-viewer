# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.log import init_logging
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.storage import init_assets_dir

from datasets_based.config import AppConfig
from datasets_based.resources import LibrariesResource
from datasets_based.worker_factory import WorkerFactory
from datasets_based.worker_loop import WorkerLoop

if __name__ == "__main__":
    app_config = AppConfig.from_env()

    init_logging(log_level=app_config.common.log_level)
    # ^ set first to have logs as soon as possible
    assets_directory = init_assets_dir(directory=app_config.assets.storage_directory)

    processing_graph = ProcessingGraph(app_config.processing_graph.specification)
    processing_step = processing_graph.get_step(app_config.datasets_based.endpoint)

    with (
        LibrariesResource(
            hf_endpoint=app_config.common.hf_endpoint,
            init_hf_datasets_cache=app_config.datasets_based.hf_datasets_cache,
            numba_path=app_config.numba.path,
        ) as libraries_resource,
        CacheMongoResource(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url),
        QueueMongoResource(database=app_config.queue.mongo_database, host=app_config.queue.mongo_url),
    ):
        worker_factory = WorkerFactory(
            app_config=app_config,
            processing_graph=processing_graph,
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
            assets_directory=assets_directory,
        )
        queue = Queue(type=processing_step.job_type, max_jobs_per_namespace=app_config.queue.max_jobs_per_namespace)
        worker_loop = WorkerLoop(
            queue=queue,
            library_cache_paths=libraries_resource.storage_paths,
            worker_factory=worker_factory,
            worker_loop_config=app_config.worker_loop,
        )
        worker_loop.loop()
