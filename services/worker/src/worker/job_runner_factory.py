# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import JobInfo
from libcommon.storage import StrPath

from worker.config import AppConfig
from worker.job_runner import JobRunner
from worker.job_runners.config.info import ConfigInfoJobRunner
from worker.job_runners.config.parquet import ConfigParquetJobRunner
from worker.job_runners.config.parquet_and_info import ConfigParquetAndInfoJobRunner
from worker.job_runners.config.size import ConfigSizeJobRunner
from worker.job_runners.config.split_names_from_dataset_info import (
    SplitNamesFromDatasetInfoJobRunner,
)
from worker.job_runners.config.split_names_from_streaming import (
    SplitNamesFromStreamingJobRunner,
)
from worker.job_runners.config_names import ConfigNamesJobRunner
from worker.job_runners.dataset.info import DatasetInfoJobRunner
from worker.job_runners.dataset.is_valid import DatasetIsValidJobRunner
from worker.job_runners.dataset.parquet import DatasetParquetJobRunner
from worker.job_runners.dataset.size import DatasetSizeJobRunner
from worker.job_runners.dataset.split_names import DatasetSplitNamesJobRunner
from worker.job_runners.dataset.split_names_from_dataset_info import (
    DatasetSplitNamesFromDatasetInfoJobRunner,
)
from worker.job_runners.dataset.split_names_from_streaming import (
    DatasetSplitNamesFromStreamingJobRunner,
)
from worker.job_runners.parquet_and_dataset_info import ParquetAndDatasetInfoJobRunner
from worker.job_runners.split.first_rows_from_parquet import (
    SplitFirstRowsFromParquetJobRunner,
)
from worker.job_runners.split.first_rows_from_streaming import (
    SplitFirstRowsFromStreamingJobRunner,
)
from worker.job_runners.split.opt_in_out_urls_scan_from_streaming import (
    SplitOptInOutUrlsScanJobRunner,
)


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
        if job_type == SplitNamesFromStreamingJobRunner.get_job_type():
            return SplitNamesFromStreamingJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                hf_datasets_cache=self.hf_datasets_cache,
            )
        if job_type == SplitFirstRowsFromStreamingJobRunner.get_job_type():
            return SplitFirstRowsFromStreamingJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                hf_datasets_cache=self.hf_datasets_cache,
                assets_directory=self.assets_directory,
            )
        if job_type == ParquetAndDatasetInfoJobRunner.get_job_type():
            return ParquetAndDatasetInfoJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                hf_datasets_cache=self.hf_datasets_cache,
            )
        if job_type == ConfigParquetAndInfoJobRunner.get_job_type():
            return ConfigParquetAndInfoJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                hf_datasets_cache=self.hf_datasets_cache,
            )
        if job_type == ConfigParquetJobRunner.get_job_type():
            return ConfigParquetJobRunner(
                job_info=job_info,
                common_config=self.app_config.common,
                worker_config=self.app_config.worker,
                processing_step=processing_step,
            )
        if job_type == DatasetParquetJobRunner.get_job_type():
            return DatasetParquetJobRunner(
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
        if job_type == ConfigInfoJobRunner.get_job_type():
            return ConfigInfoJobRunner(
                job_info=job_info,
                common_config=self.app_config.common,
                worker_config=self.app_config.worker,
                processing_step=processing_step,
            )
        if job_type == DatasetSizeJobRunner.get_job_type():
            return DatasetSizeJobRunner(
                job_info=job_info,
                common_config=self.app_config.common,
                worker_config=self.app_config.worker,
                processing_step=processing_step,
            )
        if job_type == ConfigSizeJobRunner.get_job_type():
            return ConfigSizeJobRunner(
                job_info=job_info,
                common_config=self.app_config.common,
                worker_config=self.app_config.worker,
                processing_step=processing_step,
            )
        if job_type == SplitNamesFromDatasetInfoJobRunner.get_job_type():
            return SplitNamesFromDatasetInfoJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                hf_datasets_cache=self.hf_datasets_cache,
            )
        if job_type == DatasetSplitNamesJobRunner.get_job_type():
            return DatasetSplitNamesJobRunner(
                job_info=job_info,
                processing_step=processing_step,
                common_config=self.app_config.common,
                worker_config=self.app_config.worker,
            )
        if job_type == DatasetSplitNamesFromStreamingJobRunner.get_job_type():
            return DatasetSplitNamesFromStreamingJobRunner(
                job_info=job_info,
                processing_step=processing_step,
                common_config=self.app_config.common,
                worker_config=self.app_config.worker,
            )
        if job_type == DatasetSplitNamesFromDatasetInfoJobRunner.get_job_type():
            return DatasetSplitNamesFromDatasetInfoJobRunner(
                job_info=job_info,
                processing_step=processing_step,
                common_config=self.app_config.common,
                worker_config=self.app_config.worker,
            )
        if job_type == SplitFirstRowsFromParquetJobRunner.get_job_type():
            return SplitFirstRowsFromParquetJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                hf_datasets_cache=self.hf_datasets_cache,
                assets_directory=self.assets_directory,
            )
        if job_type == DatasetIsValidJobRunner.get_job_type():
            return DatasetIsValidJobRunner(
                job_info=job_info,
                processing_step=processing_step,
                common_config=self.app_config.common,
                worker_config=self.app_config.worker,
            )

        if job_type == SplitOptInOutUrlsScanJobRunner.get_job_type():
            return SplitOptInOutUrlsScanJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                hf_datasets_cache=self.hf_datasets_cache,
            )

        supported_job_types = [
            ConfigNamesJobRunner.get_job_type(),
            SplitNamesFromStreamingJobRunner.get_job_type(),
            SplitFirstRowsFromStreamingJobRunner.get_job_type(),
            ParquetAndDatasetInfoJobRunner.get_job_type(),
            ConfigParquetAndInfoJobRunner.get_job_type(),
            ConfigParquetJobRunner.get_job_type(),
            DatasetParquetJobRunner.get_job_type(),
            DatasetInfoJobRunner.get_job_type(),
            ConfigInfoJobRunner.get_job_type(),
            DatasetSizeJobRunner.get_job_type(),
            ConfigSizeJobRunner.get_job_type(),
            SplitNamesFromDatasetInfoJobRunner.get_job_type(),
            DatasetSplitNamesFromStreamingJobRunner.get_job_type(),
            DatasetSplitNamesFromDatasetInfoJobRunner.get_job_type(),
            SplitFirstRowsFromParquetJobRunner.get_job_type(),
            DatasetIsValidJobRunner.get_job_type(),
            SplitOptInOutUrlsScanJobRunner.get_job_type(),
        ]
        raise ValueError(f"Unsupported job type: '{job_type}'. The supported job types are: {supported_job_types}")
