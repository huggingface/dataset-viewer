# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from parquet.config import WorkerConfig
from parquet.worker import ParquetWorker

if __name__ == "__main__":
    worker_config = WorkerConfig()
    ParquetWorker(worker_config).loop()
