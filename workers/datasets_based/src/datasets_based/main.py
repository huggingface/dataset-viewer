# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue
from libcommon.resources import (
    CacheDatabaseResource,
    LogResource,
    QueueDatabaseResource,
)

from datasets_based.config import AppConfig
from datasets_based.resources import LibrariesResource
from datasets_based.worker_factory import WorkerFactory
from datasets_based.worker_loop import WorkerLoop

if __name__ == "__main__":
    app_config = AppConfig.from_env()
    processing_graph = ProcessingGraph(app_config.processing_graph.specification)
    processing_step = processing_graph.get_step(app_config.datasets_based.endpoint)

    with (
        LogResource(log_level=app_config.common.log_level),
        # ^ first resource to be acquired, in order to have logs as soon as possible
        LibrariesResource(
            hf_endpoint=app_config.common.hf_endpoint,
            init_hf_datasets_cache=app_config.datasets_based.hf_datasets_cache,
            numba_path=app_config.numba.path,
        ) as libraries_resource,
        CacheDatabaseResource(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url),
        QueueDatabaseResource(database=app_config.queue.mongo_database, host=app_config.queue.mongo_url),
    ):
        worker_factory = WorkerFactory(
            app_config=app_config,
            processing_graph=processing_graph,
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )
        queue = Queue(type=processing_step.job_type, max_jobs_per_namespace=app_config.queue.max_jobs_per_namespace)
        worker_loop = WorkerLoop(
            queue=queue,
            library_cache_paths=libraries_resource.storage_paths,
            worker_factory=worker_factory,
            worker_loop_config=app_config.worker_loop,
        )
        worker_loop.loop()
