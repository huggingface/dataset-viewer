# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from libcommon.processing_graph import ProcessingGraph
from libcommon.storage import StrPath

from worker.config import AppConfig, FirstRowsConfig, ParquetAndDatasetInfoConfig
from worker.job_runner import JobInfo, JobRunner
from worker.job_runners.config_names import ConfigNamesJobRunner
from worker.job_runners.dataset_info import DatasetInfoJobRunner
from worker.job_runners.first_rows import FirstRowsJobRunner
from worker.job_runners.parquet import ParquetJobRunner
from worker.job_runners.parquet_and_dataset_info import ParquetAndDatasetInfoJobRunner
from worker.job_runners.sizes import SizesJobRunner
from worker.job_runners.split_names_streaming import SplitNamesStreamingJobRunner
from worker.job_runners.splits import SplitsJobRunner


class BaseJobRunnerFactory(ABC):
    """
    Base class for job runner factories. A job runner factory is a class that creates a job runner.

    It cannot be instantiated directly, but must be subclassed.

    Note that this class is only implemented once in the code, but we need it for the tests.
    """

    def create_job_runner(self, job_info: JobInfo) -> JobRunner:
        return self._create_job_runner(job_info=job_info)

    @abstractmethod
    def _create_job_runner(self, job_info: JobInfo) -> JobRunner:
        pass


@dataclass
class JobRunnerFactory(BaseJobRunnerFactory):
    app_config: AppConfig
    processing_graph: ProcessingGraph
    hf_datasets_cache: Path
    assets_directory: StrPath

    def _create_job_runner(self, job_info: JobInfo) -> JobRunner:
        job_type = job_info["type"]
        try:
            processing_step = self.processing_graph.get_step_by_job_type(job_type)
        except ValueError as e:
            raise ValueError(
                f"Unsupported job type: '{job_type}'. The job types declared in the processing graph are:"
                f" {[step.job_type for step in self.processing_graph.steps.values()]}"
            ) from e
        if job_type == ConfigNamesJobRunner.get_job_type():
            return ConfigNamesJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                hf_datasets_cache=self.hf_datasets_cache,
            )
        if job_type == SplitNamesStreamingJobRunner.get_job_type():
            return SplitNamesStreamingJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                hf_datasets_cache=self.hf_datasets_cache,
            )
        if job_type == SplitsJobRunner.get_job_type():
            return SplitsJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                hf_datasets_cache=self.hf_datasets_cache,
            )
        if job_type == FirstRowsJobRunner.get_job_type():
            first_rows_config = FirstRowsConfig.from_env()
            return FirstRowsJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                hf_datasets_cache=self.hf_datasets_cache,
                first_rows_config=first_rows_config,
                assets_directory=self.assets_directory,
            )
        if job_type == ParquetAndDatasetInfoJobRunner.get_job_type():
            return ParquetAndDatasetInfoJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                hf_datasets_cache=self.hf_datasets_cache,
                parquet_and_dataset_info_config=ParquetAndDatasetInfoConfig.from_env(),
            )
        if job_type == ParquetJobRunner.get_job_type():
            return ParquetJobRunner(
                job_info=job_info,
                common_config=self.app_config.common,
                worker_config=self.app_config.worker,
                processing_step=processing_step,
            )
        if job_type == DatasetInfoJobRunner.get_job_type():
            return DatasetInfoJobRunner(
                job_info=job_info,
                common_config=self.app_config.common,
                worker_config=self.app_config.worker,
                processing_step=processing_step,
            )
        if job_type == SizesJobRunner.get_job_type():
            return SizesJobRunner(
                job_info=job_info,
                common_config=self.app_config.common,
                worker_config=self.app_config.worker,
                processing_step=processing_step,
            )
        supported_job_types = [
            ConfigNamesJobRunner.get_job_type(),
            SplitNamesStreamingJobRunner.get_job_type(),
            SplitsJobRunner.get_job_type(),
            FirstRowsJobRunner.get_job_type(),
            ParquetAndDatasetInfoJobRunner.get_job_type(),
            ParquetJobRunner.get_job_type(),
            DatasetInfoJobRunner.get_job_type(),
            SizesJobRunner.get_job_type(),
        ]
        raise ValueError(f"Unsupported job type: '{job_type}'. The supported job types are: {supported_job_types}")
