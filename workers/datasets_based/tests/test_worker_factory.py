# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

import pytest
from libcommon.queue import Priority

from datasets_based.config import AppConfig
from datasets_based.worker import JobInfo
from datasets_based.worker_factory import DatasetBasedWorkerFactory


@pytest.mark.parametrize(
    "job_type,expected_worker",
    [
        ("/config-names", "ConfigNamesWorker"),
        ("/splits", "SplitsWorker"),
        ("/first-rows", "FirstRowsWorker"),
        ("/parquet-and-dataset-info", "ParquetAndDatasetInfoWorker"),
        ("/parquet", "ParquetWorker"),
        ("/dataset-info", "DatasetInfoWorker"),
        ("/sizes", "SizesWorker"),
        ("/unknown", None),
    ],
)
def test_create_worker(app_config: AppConfig, job_type: str, expected_worker: Optional[str]) -> None:
    worker_factory = DatasetBasedWorkerFactory(app_config=app_config)
    job_info: JobInfo = {
        "type": job_type,
        "dataset": "dataset",
        "config": "config",
        "split": "split",
        "job_id": "job_id",
        "force": False,
        "priority": Priority.NORMAL,
    }
    if expected_worker is None:
        with pytest.raises(ValueError):
            worker_factory.create_worker(job_info=job_info)
    else:
        worker = worker_factory.create_worker(job_info=job_info)
        assert worker.__class__.__name__ == expected_worker
