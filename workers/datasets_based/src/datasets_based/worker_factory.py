# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from abc import ABC, abstractmethod

from datasets_based.config import (
    AppConfig,
    FirstRowsConfig,
    ParquetAndDatasetInfoConfig,
)
from datasets_based.worker import JobInfo, Worker
from datasets_based.workers.basic_stats import BasicStatsWorker
from datasets_based.worker import JobInfo, Worker
from datasets_based.workers.config_names import ConfigNamesWorker
from datasets_based.workers.dataset_info import DatasetInfoWorker
from datasets_based.workers.first_rows import FirstRowsWorker
from datasets_based.workers.parquet import ParquetWorker
from datasets_based.workers.parquet_and_dataset_info import ParquetAndDatasetInfoWorker
from datasets_based.workers.sizes import SizesWorker
from datasets_based.workers.split_names import SplitNamesWorker
from datasets_based.workers.splits import SplitsWorker


class BaseWorkerFactory(ABC):
    """
    Base class for worker factories. A worker factory is a class that creates a worker.

    It cannot be instantiated directly, but must be subclassed.
    """

    def create_worker(self, job_info: JobInfo) -> Worker:
        return self._create_worker(job_info=job_info)

    @abstractmethod
    def _create_worker(self, job_info: JobInfo) -> Worker:
        pass


class WorkerFactory(BaseWorkerFactory):
    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config

    def _create_worker(self, job_info: JobInfo) -> Worker:
        job_type = job_info["type"]
        try:
            processing_step = self.app_config.processing_graph.graph.get_step_by_job_type(job_type)
        except ValueError as e:
            raise ValueError(
                f"Unsupported job type: '{job_type}'. The job types declared in the processing graph are:"
                f" {[step.job_type for step in self.app_config.processing_graph.graph.steps.values()]}"
            ) from e
        if job_type == ConfigNamesWorker.get_job_type():
            return ConfigNamesWorker(job_info=job_info, app_config=self.app_config, processing_step=processing_step)
        if job_type == SplitNamesWorker.get_job_type():
            return SplitNamesWorker(job_info=job_info, app_config=self.app_config, processing_step=processing_step)
        if job_type == SplitsWorker.get_job_type():
            return SplitsWorker(job_info=job_info, app_config=self.app_config, processing_step=processing_step)
        if job_type == FirstRowsWorker.get_job_type():
            first_rows_config = FirstRowsConfig.from_env()
            return FirstRowsWorker(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                first_rows_config=first_rows_config,
            )
        if job_type == ParquetAndDatasetInfoWorker.get_job_type():
            return ParquetAndDatasetInfoWorker(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                parquet_and_dataset_info_config=ParquetAndDatasetInfoConfig.from_env(),
            )
        if job_type == ParquetWorker.get_job_type():
            return ParquetWorker(
                job_info=job_info, common_config=self.app_config.common, processing_step=processing_step
            )
        if job_type == DatasetInfoWorker.get_job_type():
            return DatasetInfoWorker(
                job_info=job_info, common_config=self.app_config.common, processing_step=processing_step
            )
        if job_type == SizesWorker.get_job_type():
            return SizesWorker(
                job_info=job_info, common_config=self.app_config.common, processing_step=processing_step
            )
        if job_type == BasicStatsWorker.get_job_type():
            return BasicStatsWorker(job_info=job_info, app_config=self.app_config, processing_step=processing_step
            )
        supported_job_types = [
            ConfigNamesWorker.get_job_type(),
            SplitNamesWorker.get_job_type(),
            SplitsWorker.get_job_type(),
            FirstRowsWorker.get_job_type(),
            ParquetAndDatasetInfoWorker.get_job_type(),
            ParquetWorker.get_job_type(),
            DatasetInfoWorker.get_job_type(),
            SizesWorker.get_job_type(),
            BasicStatsWorker.get_job_type(),
        ]
        raise ValueError(f"Unsupported job type: '{job_type}'. The supported job types are: {supported_job_types}")
