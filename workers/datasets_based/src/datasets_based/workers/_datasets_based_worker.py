# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import importlib.metadata
from abc import ABC, abstractmethod

from libcommon.worker import Worker

from datasets_based.config import AppConfig, DatasetsBasedConfig


class DatasetsBasedWorker(Worker, ABC):
    """Base class for workers that use datasets."""

    datasets_based_config: DatasetsBasedConfig

    @staticmethod
    @abstractmethod
    def get_endpoint() -> str:
        pass

    def __init__(self, app_config: AppConfig):
        super().__init__(
            processing_step=app_config.processing_graph.graph.get_step(self.get_endpoint()),
            # ^ raises if the step is not found
            common_config=app_config.common,
            queue_config=app_config.queue,
            worker_config=app_config.worker,
            version=importlib.metadata.version(__package__.split(".")[0]),
        )
        self.datasets_based_config = app_config.datasets_based
