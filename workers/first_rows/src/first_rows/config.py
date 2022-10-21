# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import datasets.config
from datasets.utils.logging import log_levels, set_verbosity
from environs import Env
from libcache.config import CacheConfig
from libcommon.config import CommonConfig
from libqueue.config import QueueConfig


class FirstRowsConfig:
    fallback_max_dataset_size: int
    max_bytes: int
    max_number: int
    min_cell_bytes: int
    min_number: int

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("FIRST_ROWS_"):
            self.fallback_max_dataset_size = env.int(name="FALLBACK_MAX_DATASET_SIZE", default=100_000_000)
            self.max_bytes = env.int(name="MAX_BYTES", default=1_000_000)
            self.max_number = env.int(name="MAX_NUMBER", default=100)
            self.min_cell_bytes = env.int(name="CELL_MIN_BYTES", default=100)
            self.min_number = env.int(name="MIN_NUMBER", default=10)


class WorkerConfig:
    cache: CacheConfig
    common: CommonConfig
    first_rows: FirstRowsConfig
    queue: QueueConfig

    def __init__(self):
        # First process the common configuration to setup the logging
        self.common = CommonConfig()
        self.cache = CacheConfig()
        self.first_rows = FirstRowsConfig()
        self.queue = QueueConfig()
        self.setup()

    def setup(self):
        # Ensure the datasets library uses the expected HuggingFace endpoint
        datasets.config.HF_ENDPOINT = self.common.hf_endpoint
        datasets.config.HUB_DATASETS_URL = self.common.hf_endpoint + "/datasets/{repo_id}/resolve/{revision}/{path}"
        # Don't increase the datasets download counts on huggingface.co
        datasets.config.HF_UPDATE_DOWNLOAD_COUNTS = False
        # Set logs from the datasets library to the least verbose
        set_verbosity(log_levels["critical"])
