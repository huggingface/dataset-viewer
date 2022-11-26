# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import datasets.config
from datasets.utils.logging import log_levels, set_verbosity
from libcommon.config import (
    CacheConfig,
    CommonConfig,
    ProcessingGraphConfig,
    QueueConfig,
    WorkerConfig,
)


class AppConfig:
    cache: CacheConfig
    common: CommonConfig
    processing_graph: ProcessingGraphConfig
    queue: QueueConfig
    worker: WorkerConfig

    def __init__(self):
        # First process the common configuration to setup the logging
        self.common = CommonConfig()
        self.cache = CacheConfig()
        self.queue = QueueConfig()
        self.processing_graph = ProcessingGraphConfig()
        self.worker = WorkerConfig()
        self.setup()

    def setup(self):
        # Ensure the datasets library uses the expected HuggingFace endpoint
        datasets.config.HF_ENDPOINT = self.common.hf_endpoint
        # Don't increase the datasets download counts on huggingface.co
        datasets.config.HF_UPDATE_DOWNLOAD_COUNTS = False
        # Set logs from the datasets library to the least verbose
        set_verbosity(log_levels["critical"])

        # Note: self.common.hf_endpoint is ignored by the huggingface_hub library for now (see
        # the discussion at https://github.com/huggingface/datasets/pull/5196), and this breaks
        # various of the datasets functions. The fix, for now, is to set the HF_ENDPOINT
        # environment variable to the desired value.
