# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import datasets.config
from datasets.utils.logging import log_levels, set_verbosity
from environs import Env
from libcache.config import CacheConfig
from libcommon.config import CommonConfig
from libqueue.config import QueueConfig


class ParquetConfig:
    commit_message: str
    source_revision: str
    target_revision: str
    url_template: str

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("PARQUET_"):
            self.commit_message = env.str(name="COMMIT_MESSAGE", default="Update parquet files")
            self.source_revision = env.str(name="SOURCE_REVISION", default="main")
            self.target_revision = env.str(name="TARGET_REVISION", default="refs/convert/parquet")
            self.url_template = env.str(
                name="URL_TEMPLATE", default="/datasets/{repo_id}/resolve/{revision}/{filename}"
            )


class WorkerConfig:
    cache: CacheConfig
    common: CommonConfig
    parquet: ParquetConfig
    queue: QueueConfig

    def __init__(self):
        # First process the common configuration to setup the logging
        self.common = CommonConfig()
        self.cache = CacheConfig()
        self.queue = QueueConfig()
        self.parquet = ParquetConfig()
        self.setup()

    def setup(self):
        # Ensure the datasets library uses the expected HuggingFace endpoint
        datasets.config.HF_ENDPOINT = self.common.hf_endpoint
        datasets.config.HUB_DATASETS_URL = self.common.hf_endpoint + "/datasets/{repo_id}/resolve/{revision}/{path}"
        # Don't increase the datasets download counts on huggingface.co
        datasets.config.HF_UPDATE_DOWNLOAD_COUNTS = False
        # Set logs from the datasets library to the least verbose
        set_verbosity(log_levels["critical"])
