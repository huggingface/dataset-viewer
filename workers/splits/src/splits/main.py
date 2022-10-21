# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from splits.config import WorkerConfig
from splits.worker import SplitsWorker

if __name__ == "__main__":
    worker_config = WorkerConfig()
    SplitsWorker(worker_config).loop()
