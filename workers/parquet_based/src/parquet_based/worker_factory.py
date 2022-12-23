# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.worker import JobInfo, Worker, WorkerFactory

from parquet_based.config import AppConfig
from parquet_based.workers.size import SizeWorker


class ParquetBasedWorkerFactory(WorkerFactory):
    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config

    def _create_worker(self, job_info: JobInfo) -> Worker:
        job_type = job_info["type"]
        if job_type == SizeWorker.get_job_type():
            return SizeWorker(job_info=job_info, app_config=self.app_config)
        else:
            supported_job_types = [SizeWorker.get_job_type()]
            raise ValueError(f"Unsupported job type: '{job_type}'. The supported job types are: {supported_job_types}")
