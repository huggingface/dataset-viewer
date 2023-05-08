# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from libcommon.processing_graph import ProcessingGraph
from libcommon.storage import StrPath
from libcommon.utils import JobInfo

from worker.config import AppConfig
from worker.job_operator import JobOperator
from worker.job_operators.config.info import ConfigInfoJobOperator
from worker.job_operators.config.opt_in_out_urls_count import (
    ConfigOptInOutUrlsCountJobOperator,
)
from worker.job_operators.config.parquet import ConfigParquetJobOperator
from worker.job_operators.config.parquet_and_info import ConfigParquetAndInfoJobOperator
from worker.job_operators.config.size import ConfigSizeJobOperator
from worker.job_operators.config.split_names_from_dataset_info import (
    SplitNamesFromDatasetInfoJobOperator,
)
from worker.job_operators.config.split_names_from_streaming import (
    SplitNamesFromStreamingJobOperator,
)
from worker.job_operators.dataset.config_names import ConfigNamesJobOperator
from worker.job_operators.dataset.info import DatasetInfoJobOperator
from worker.job_operators.dataset.is_valid import DatasetIsValidJobOperator
from worker.job_operators.dataset.opt_in_out_urls_count import (
    DatasetOptInOutUrlsCountJobOperator,
)
from worker.job_operators.dataset.parquet import DatasetParquetJobOperator
from worker.job_operators.dataset.size import DatasetSizeJobOperator
from worker.job_operators.dataset.split_names import DatasetSplitNamesJobOperator
from worker.job_operators.split.first_rows_from_parquet import (
    SplitFirstRowsFromParquetJobOperator,
)
from worker.job_operators.split.first_rows_from_streaming import (
    SplitFirstRowsFromStreamingJobOperator,
)
from worker.job_operators.split.opt_in_out_urls_count import (
    SplitOptInOutUrlsCountJobOperator,
)
from worker.job_operators.split.opt_in_out_urls_scan_from_streaming import (
    SplitOptInOutUrlsScanJobOperator,
)


class BaseJobOperatorFactory(ABC):
    """
    Base class for job runner factories. A job runner factory is a class that creates a job runner.

    It cannot be instantiated directly, but must be subclassed.

    Note that this class is only implemented once in the code, but we need it for the tests.
    """

    def create_job_runner(self, job_info: JobInfo) -> JobOperator:
        return self._create_job_runner(job_info=job_info)

    @abstractmethod
    def _create_job_runner(self, job_info: JobInfo) -> JobOperator:
        pass


@dataclass
class JobOperatorFactory(BaseJobOperatorFactory):
    app_config: AppConfig
    processing_graph: ProcessingGraph
    hf_datasets_cache: Path
    assets_directory: StrPath

    def _create_job_runner(self, job_info: JobInfo) -> JobOperator:
        job_type = job_info["type"]
        try:
            processing_step = self.processing_graph.get_step_by_job_type(job_type)
        except ValueError as e:
            raise ValueError(
                f"Unsupported job type: '{job_type}'. The job types declared in the processing graph are:"
                f" {[step.job_type for step in self.processing_graph.steps.values()]}"
            ) from e
        if job_type == ConfigNamesJobOperator.get_job_type():
            return ConfigNamesJobOperator(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                hf_datasets_cache=self.hf_datasets_cache,
            )
        if job_type == SplitNamesFromStreamingJobOperator.get_job_type():
            return SplitNamesFromStreamingJobOperator(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                hf_datasets_cache=self.hf_datasets_cache,
            )
        if job_type == SplitFirstRowsFromStreamingJobOperator.get_job_type():
            return SplitFirstRowsFromStreamingJobOperator(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                hf_datasets_cache=self.hf_datasets_cache,
                assets_directory=self.assets_directory,
            )
        if job_type == ConfigParquetAndInfoJobOperator.get_job_type():
            return ConfigParquetAndInfoJobOperator(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                hf_datasets_cache=self.hf_datasets_cache,
            )
        if job_type == ConfigParquetJobOperator.get_job_type():
            return ConfigParquetJobOperator(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
            )
        if job_type == DatasetParquetJobOperator.get_job_type():
            return DatasetParquetJobOperator(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
            )
        if job_type == DatasetInfoJobOperator.get_job_type():
            return DatasetInfoJobOperator(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
            )
        if job_type == ConfigInfoJobOperator.get_job_type():
            return ConfigInfoJobOperator(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
            )
        if job_type == DatasetSizeJobOperator.get_job_type():
            return DatasetSizeJobOperator(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
            )
        if job_type == ConfigSizeJobOperator.get_job_type():
            return ConfigSizeJobOperator(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
            )
        if job_type == SplitNamesFromDatasetInfoJobOperator.get_job_type():
            return SplitNamesFromDatasetInfoJobOperator(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
            )
        if job_type == DatasetSplitNamesJobOperator.get_job_type():
            return DatasetSplitNamesJobOperator(
                job_info=job_info,
                processing_step=processing_step,
                app_config=self.app_config,
            )
        if job_type == SplitFirstRowsFromParquetJobOperator.get_job_type():
            return SplitFirstRowsFromParquetJobOperator(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                assets_directory=self.assets_directory,
            )
        if job_type == DatasetIsValidJobOperator.get_job_type():
            return DatasetIsValidJobOperator(
                job_info=job_info,
                processing_step=processing_step,
                app_config=self.app_config,
            )

        if job_type == SplitOptInOutUrlsScanJobOperator.get_job_type():
            return SplitOptInOutUrlsScanJobOperator(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
                hf_datasets_cache=self.hf_datasets_cache,
            )
        if job_type == ConfigOptInOutUrlsCountJobOperator.get_job_type():
            return ConfigOptInOutUrlsCountJobOperator(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
            )
        if job_type == DatasetOptInOutUrlsCountJobOperator.get_job_type():
            return DatasetOptInOutUrlsCountJobOperator(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
            )

        if job_type == SplitOptInOutUrlsCountJobOperator.get_job_type():
            return SplitOptInOutUrlsCountJobOperator(
                job_info=job_info,
                app_config=self.app_config,
                processing_step=processing_step,
            )

        supported_job_types = [
            ConfigNamesJobOperator.get_job_type(),
            SplitNamesFromStreamingJobOperator.get_job_type(),
            SplitFirstRowsFromStreamingJobOperator.get_job_type(),
            ConfigParquetAndInfoJobOperator.get_job_type(),
            ConfigParquetJobOperator.get_job_type(),
            DatasetParquetJobOperator.get_job_type(),
            DatasetInfoJobOperator.get_job_type(),
            ConfigInfoJobOperator.get_job_type(),
            DatasetSizeJobOperator.get_job_type(),
            ConfigSizeJobOperator.get_job_type(),
            SplitNamesFromDatasetInfoJobOperator.get_job_type(),
            SplitFirstRowsFromParquetJobOperator.get_job_type(),
            DatasetIsValidJobOperator.get_job_type(),
            SplitOptInOutUrlsScanJobOperator.get_job_type(),
            SplitOptInOutUrlsCountJobOperator.get_job_type(),
            ConfigOptInOutUrlsCountJobOperator.get_job_type(),
            DatasetOptInOutUrlsCountJobOperator.get_job_type(),
        ]
        raise ValueError(f"Unsupported job type: '{job_type}'. The supported job types are: {supported_job_types}")
