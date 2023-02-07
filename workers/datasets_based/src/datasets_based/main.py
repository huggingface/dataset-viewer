# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.queue import Queue

from datasets_based.config import AppConfig
from datasets_based.worker_factory import WorkerFactory
from datasets_based.worker_loop import WorkerLoop

if __name__ == "__main__":
    app_config = AppConfig.from_env()
    processing_graph = app_config.processing_graph.graph
    processing_step = processing_graph.get_step(app_config.datasets_based.endpoint)
    worker_factory = WorkerFactory(app_config=app_config, processing_graph=processing_graph)
    queue = Queue(type=processing_step.job_type, max_jobs_per_namespace=app_config.queue.max_jobs_per_namespace)
    worker_loop = WorkerLoop(
        queue=queue,
        worker_factory=worker_factory,
        worker_loop_config=app_config.worker_loop,
    )
    worker_loop.loop()
