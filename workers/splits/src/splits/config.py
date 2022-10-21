# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import datasets.config
from datasets.utils.logging import log_levels, set_verbosity
from libcache.config import CacheConfig
from libcommon.config import CommonConfig
from libqueue.config import QueueConfig


class WorkerConfig:
    cache: CacheConfig
    common: CommonConfig
    queue: QueueConfig

    def __init__(self):
        self.cache = CacheConfig()
        self.common = CommonConfig()
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
