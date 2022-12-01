# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from datasets_based.config import AppConfig
from datasets_based.worker import get_worker

if __name__ == "__main__":
    app_config = AppConfig()
    worker = get_worker(app_config)
    worker.loop()
