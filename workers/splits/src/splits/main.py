# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from splits.config import AppConfig
from splits.worker import SplitsWorker

if __name__ == "__main__":
    app_config = AppConfig()
    SPLITS_ENDPOINT = "/splits"
    SplitsWorker(app_config=app_config, endpoint=SPLITS_ENDPOINT).loop()
