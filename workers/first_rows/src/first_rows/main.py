# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from first_rows.config import AppConfig
from first_rows.worker import FirstRowsWorker

if __name__ == "__main__":
    app_config = AppConfig()
    FIRST_ROWS_ENDPOINT = "/first-rows"
    FirstRowsWorker(app_config=app_config, endpoint=FIRST_ROWS_ENDPOINT).loop()
