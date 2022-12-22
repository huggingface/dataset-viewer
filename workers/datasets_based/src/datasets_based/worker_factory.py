# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.worker import JobInfo, Worker, WorkerFactory

from datasets_based.config import AppConfig, FirstRowsConfig, ParquetConfig
from datasets_based.workers.first_rows import FirstRowsWorker
from datasets_based.workers.parquet import ParquetWorker
from datasets_based.workers.splits import SplitsWorker


class DatasetBasedWorkerFactory(WorkerFactory):
    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config

    def _create_worker(self, job_info: JobInfo) -> Worker:
        job_type = job_info["type"]
        if job_type == SplitsWorker.get_job_type():
            return SplitsWorker(job_info=job_info, app_config=self.app_config)
        elif job_type == FirstRowsWorker.get_job_type():
            return FirstRowsWorker(
                job_info=job_info, app_config=self.app_config, first_rows_config=FirstRowsConfig.from_env()
            )
        elif job_type == ParquetWorker.get_job_type():
            return ParquetWorker(
                job_info=job_info, app_config=self.app_config, parquet_config=ParquetConfig.from_env()
            )
        else:
            supported_job_types = [
                SplitsWorker.get_job_type(),
                FirstRowsWorker.get_job_type(),
                ParquetWorker.get_job_type(),
            ]
            raise ValueError(f"Unsupported job type: '{job_type}'. The supported job types are: {supported_job_types}")
