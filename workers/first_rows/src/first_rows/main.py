# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from first_rows.config import WorkerConfig
from first_rows.worker import FirstRowsWorker

if __name__ == "__main__":
    worker_config = WorkerConfig()
    FirstRowsWorker(worker_config).loop()
