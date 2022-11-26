# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from parquet.config import AppConfig
from parquet.worker import ParquetWorker

if __name__ == "__main__":
    app_config = AppConfig()
    PARQUET_ENDPOINT = "/parquet"
    ParquetWorker(app_config=app_config, endpoint=PARQUET_ENDPOINT).loop()
